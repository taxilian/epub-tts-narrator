"""Explain speech cache reuse without generating audio or changing saved files."""

from collections import Counter, defaultdict
from pathlib import Path

from .book_context import resolve_book_config, voice_store
from .bundle import chunk_speakers
from .common import file_hash, read_json, regular_file
from .speech_cache import SpeechCache
from .speech_identity import engines_compatible, synthesis_engine_identity, _model_alias
from .workflow import make_render_plan, _paragraph_request, _speech_request


class _ReadOnlySpeechCache(SpeechCache):
    def _publish(self, key, entry):
        self._register(key, entry)


def _locations(request):
    chunk = request.get("chunk", {})
    result = defaultdict(list)
    for part in chunk.get("paragraph_turns", [chunk]):
        row = part.get("segment", {})
        key = (row.get("chapter_id"), row.get("block_id"))
        if all(isinstance(value, str) for value in key):
            result[key].extend((span.get("start"), span.get("end"))
                              for span in part.get("source_ranges", [])
                              if type(span.get("start")) is int and type(span.get("end")) is int)
    return dict(result)


def _overlap(current, saved):
    score = 0
    for key in current.keys() & saved.keys():
        score += sum(max(0, min(end, other_end) - max(start, other_start))
                     for start, end in current[key] for other_start, other_end in saved[key])
    return score


def _fields(saved, current, prefix=""):
    if isinstance(saved, dict) and isinstance(current, dict):
        return [field for key in sorted(saved.keys() | current.keys())
                for field in _fields(saved.get(key), current.get(key), f"{prefix}.{key}".lstrip("."))]
    if isinstance(saved, list) and isinstance(current, list) and len(saved) == len(current):
        return [field for i, (before, after) in enumerate(zip(saved, current))
                for field in _fields(before, after, f"{prefix}[{i}]")]
    return [prefix] if saved != current else []


def _identity_issue(exc):
    message = str(exc)
    if "voice" in message.lower() or "reference" in message.lower():
        return "unresolved_saved_voice"
    if "Recorded synthesis text" in message:
        return "recorded_synthesis_text_changed"
    if "Recorded delivery" in message:
        return "recorded_delivery_changed"
    if "Recorded dialogue grouping" in message:
        return "recorded_grouping_changed"
    return "invalid_saved_speech_identity"


def _inventory(cache):
    """Retain misses too, so missing WAVs and unresolved voices are explainable."""
    cache._load()  # Use exactly the renderer's integrity and import rules.
    indexed = defaultdict(list)
    for entry in cache.entries.values():
        indexed[entry["request_hash"]].append(entry)
    disk_indexes = defaultdict(list)
    for path in sorted((cache.work / "speech-cache").glob("*.json")):
        entry = cache._read_index(path)
        disk_indexes[entry["request_hash"]].append(entry)
    records, counts = [], Counter()
    for path in sorted((cache.work / "clips").glob("*/accepted.json")):
        raw = read_json(regular_file(path))
        saved = cache._accepted(path.parent.name)
        if (not isinstance(raw, dict) or not isinstance(raw.get("report"), dict)
                or raw["report"].get("passed") is not True):
            counts["rejected_records"] += 1
            continue
        request = raw["request"]
        if request.get("kind") not in {"speech", "paragraph_speech"}:
            continue
        counts["accepted_records"] += 1
        issue, identity = None, None
        if saved is None:
            issue = "missing_audio"
        elif disk_indexes[path.parent.name] and not any(
                entry["accepted_sha256"] == file_hash(path) for entry in disk_indexes[path.parent.name]):
            issue = "acceptance_changed_since_indexing"
        else:
            try:
                identity = cache.identity(request, engine_metadata=raw.get("engine_metadata"))
            except ValueError as exc:
                entries = indexed[path.parent.name]
                identity = next((entry["identity"] for entry in entries
                                 if entry["accepted_sha256"] == file_hash(path)), None)
                if identity is None:
                    issue = _identity_issue(exc)
        if issue:
            counts[issue] += 1
        records.append({"request": request, "request_hash": path.parent.name,
                        "identity": identity, "issue": issue,
                        "engine_metadata": raw.get("engine_metadata"),
                        "locations": _locations(request)})
    counts["index_records"] = sum(map(len, disk_indexes.values()))
    return records, counts


def _engine_differences(saved, current):
    # Compare the same policy shape on both sides. An alias intentionally drops
    # model weights and nests common sampling; that layout is not a change to
    # temperature or token limits in the actual synthesis request.
    alias = _model_alias(current)
    if alias is not None:
        before = synthesis_engine_identity(dict(saved,
            cache_model_alias=_model_alias(saved) or saved.get("model")))
        after = synthesis_engine_identity(current)
        if before.get("engine") != after.get("engine"):
            return ["engine.compatibility"]
    else:
        before, after = dict(saved), dict(current)
        before.pop("cache_model_alias", None)
        after.pop("cache_model_alias", None)
        before, after = synthesis_engine_identity(before), synthesis_engine_identity(after)
    return _fields(before, after, "engine") or ["engine.compatibility"]


def _differences(saved, identity, request):
    if saved["issue"]:
        return [saved["issue"]]
    before = saved["identity"]
    changed = _fields({k: v for k, v in before.items() if k != "engine"},
                      {k: v for k, v in identity.items() if k != "engine"})
    if not engines_compatible(saved["request"]["engine"], request["engine"],
                              engine_metadata=saved["engine_metadata"]):
        changed.extend(_engine_differences(saved["request"]["engine"], request["engine"]))
    if "kind" in changed:
        changed[changed.index("kind")] = "request_kind_or_grouping"
    if "text" in changed:
        changed[changed.index("text")] = "text_or_delivery"
    return changed or ["saved_record_not_reusable"]


def cache_report(bundle, config, engine_identity, *, multi_speaker=True, max_examples=10):
    """Report actual cache hits and the nearest prior inputs for current misses.

    Candidate comparisons explain differences; they never authorize reuse. The
    normal speech cache alone decides hits, including successful verification
    provenance and waveform checksums. Run while rendering is paused for a stable
    snapshot. Text, voice transcripts, credentials and raw engine metadata are
    deliberately omitted from the shareable report.
    """
    if type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_examples must be a nonnegative integer")
    config = resolve_book_config(bundle, config)
    plan = make_render_plan(bundle, config, multi_speaker=multi_speaker)
    if plan["errors"]:
        raise ValueError("\n".join(plan["errors"]))
    store = voice_store(bundle, config)
    voices = {voice_id: store.load(voice_id) for voice_id in plan["voice_ids"]}
    available = {key: value for key, value in voices.items() if value is not None}
    cache = _ReadOnlySpeechCache(config["render"]["work_dir"], available)
    saved, inventory = _inventory(cache)
    by_block, by_text = defaultdict(list), defaultdict(list)
    for index, item in enumerate(saved):
        for location in item["locations"]:
            by_block[location].append(index)
        if item["identity"] and isinstance(item["identity"].get("text"), str):
            by_text[item["identity"]["text"]].append(index)
    totals = Counter(requests=0, hits=0, misses=0, missing_voice_requests=0,
                     accepted_records=0, index_records=0)
    totals.update(inventory)
    reasons, examples = Counter(), []
    for chapter in plan["chapters"]:
        for chunk in chapter["chunks"]:
            if chunk["silent_control"]:
                continue
            speakers = chunk_speakers(chunk)
            groups = [speakers] if "paragraph_turns" in chunk else [[speaker] for speaker in speakers]
            for group in groups:
                totals["requests"] += 1
                missing = [speaker["voice_id"] for speaker in group if voices[speaker["voice_id"]] is None]
                example = {"chapter_number": chapter["number"], "chunk_id": chunk["chunk_id"],
                           "block_ids": sorted({turn["segment"]["block_id"]
                              for turn in chunk.get("paragraph_turns", [chunk])})}
                if missing:
                    totals["missing_voice_requests"] += 1
                    differences = ["missing_current_voice"]
                    example["missing_voice_ids"] = missing
                else:
                    request = (_paragraph_request(chunk, available, engine_identity, config)
                               if "paragraph_turns" in chunk else
                               _speech_request(chunk, available[group[0]["voice_id"]], engine_identity,
                                               config, performer=group[0] if len(speakers) > 1 else None))
                    identity = cache.identity(request)
                    if cache.find(identity, request=request) is not None:
                        totals["hits"] += 1
                        continue
                    locations = _locations(request)
                    candidates = {index for location in locations for index in by_block[location]}
                    candidates.update(by_text.get(identity.get("text"), []))
                    ranked = []
                    for index in candidates:
                        item = saved[index]
                        differences = _differences(item, identity, request)
                        same_text = item["identity"] is not None and (
                            item["identity"].get("text") == identity.get("text"))
                        ranked.append((not same_text, -_overlap(locations, item["locations"]),
                                       len(differences), item["request_hash"], differences, item))
                    if ranked:
                        _, _, _, _, differences, nearest = min(ranked, key=lambda item: item[:4])
                        example["saved_request_hash"] = nearest["request_hash"]
                        # Only deliberately safe engine labels, never arbitrary metadata.
                        example["models"] = {
                            "saved": nearest["request"]["engine"].get("model"),
                            "current": request["engine"].get("model")}
                    else:
                        differences = ["no_saved_source_or_text_match"]
                totals["misses"] += 1
                reasons.update(differences)
                if len(examples) < max_examples:
                    examples.append(dict(example, differences=differences))
    return {"format": "audiobook-cache-report", "version": 1,
            "book_directory": str(Path(bundle["directory"]).resolve()),
            "work_directory": str(cache.work), "summary": dict(totals),
            "reasons": dict(sorted(reasons.items())), "examples": examples,
            "examples_omitted": max(0, totals["misses"] - len(examples))}
