"""Frozen equal-work corpus for the chapter pipeline benchmark.

``prepare`` selects unique single-voice clips from the real render plan across
four specified source chapters, stratifies short/medium/long lengths, splits
them into four balanced benchmark-only chapter groups with a synthetic
benchmark identity, and packs portable lossless FLAC voice references. Every
corpus file is private (0600) because cases carry source passages, and the
original book is never modified. Loading re-verifies checksums, reference
bytes, group balance, and the frozen input hash, failing closed.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, regular_file, write_json
from epub_tts_narrator.renderer.chapter_protocol import SAFE_RENDER_KEYS
from scripts.benchmark_rendering import length_class

CORPUS_FORMAT = "epub-tts-chapter-pipeline-benchmark"
GROUP_COUNT = 4
DEFAULT_CASES = 160
_WORDS = re.compile(r"\S+")
_CASE_ID = re.compile(r"[0-9a-f]{24}\Z")


def canonical_words(text):
    """Plan-defined canonical source word count; a source span counts once."""
    return len(_WORDS.findall(text))


def _case_id(origin, chunk_id):
    return digest({"chapter": origin, "chunk": chunk_id})[:24]


def candidate_cases(plan_chapters, chapter_numbers):
    """Eligible single-voice spoken clips from the real plan, in source order."""
    by_number = {chapter["number"]: chapter for chapter in plan_chapters}
    if len(set(chapter_numbers)) != GROUP_COUNT or any(number not in by_number for number in chapter_numbers):
        raise ValueError("Select exactly four distinct source chapters; chapter numbers outside the "
                         "bundle render plan are rejected")
    from epub_tts_narrator.renderer.bundle import chunk_speakers
    candidates = []
    for number in chapter_numbers:
        for chunk in by_number[number]["chunks"]:
            if chunk["silent_control"]:
                continue
            turns = chunk.get("paragraph_turns", [chunk])
            speakers = [chunk_speakers(turn) for turn in turns]
            if any(len(entry) != 1 for entry in speakers):
                continue
            voice_ids = {entry[0]["voice_id"] for entry in speakers}
            if len(voice_ids) != 1:
                continue
            candidates.append({"origin": number, "origin_title": by_number[number]["title"],
                               "chunk": deepcopy(chunk), "voice_id": voice_ids.pop(),
                               "words": canonical_words(chunk["spoken_text"]),
                               "length_class": length_class(chunk["spoken_text"])})
    return candidates


def _interleave(candidates, chapter_numbers):
    lanes = {number: [entry for entry in candidates if entry["origin"] == number]
             for number in chapter_numbers}
    order, position = [], 0
    while any(lanes.values()):
        number = chapter_numbers[position % GROUP_COUNT]
        if lanes[number]:
            order.append(lanes[number].pop(0))
        position += 1
    return order


def _class_targets(total):
    """Fixed proportions whose counts are each divisible across the four groups."""
    long_count = GROUP_COUNT * round(total / 20)
    medium_count = GROUP_COUNT * round(total / 12)
    return {"long": long_count, "medium": medium_count, "short": total - long_count - medium_count}


def select_cases(plan_chapters, voices, chapter_numbers, *, total, seed):
    """Freeze equal unique work: balanced groups plus per-voice warmup clips."""
    if type(total) is not int or total < GROUP_COUNT or total % GROUP_COUNT:
        raise ValueError(f"Cases must be a positive multiple of {GROUP_COUNT} so groups stay balanced")
    candidates = candidate_cases(plan_chapters, chapter_numbers)
    order = _interleave(candidates, chapter_numbers)
    chosen = []
    for class_index, (name, count) in enumerate(_class_targets(total).items()):
        pool = [entry for entry in order if entry["length_class"] == name]
        if len(pool) < count:
            raise ValueError(f"Insufficient unique {name} clips in the selected chapters: "
                             f"need {count}, found {len(pool)}; choose different chapters or fewer cases")
        if count:
            # Rotating each class by a different chapter offset mixes origins
            # inside every group instead of aligning a group with one chapter.
            offset = class_index % len(pool)
            chosen.extend((pool[offset:] + pool[:offset])[:count])
    remaining = [entry for entry in order if entry not in chosen]
    warmup = []
    for asset in sorted({voices[entry["voice_id"]]["asset_hash"] for entry in chosen}):
        pick = next((entry for entry in remaining
                     if voices[entry["voice_id"]]["asset_hash"] == asset), None)
        if pick is None:
            raise ValueError("Need one extra unused clip per benchmark voice for warmup; "
                             "the selected chapters are too small for this case count")
        warmup.append(pick)
        remaining.remove(pick)
    cases = [_case_entry(entry, index % GROUP_COUNT, voices, seed) for index, entry in enumerate(chosen)]
    _require_unique(cases)
    return cases, [_case_entry(entry, None, voices, seed) for entry in warmup]


def _case_entry(candidate, group, voices, seed):
    entry = {"id": _case_id(candidate["origin"], candidate["chunk"]["chunk_id"]), "group": group,
             "origin": candidate["origin"], "origin_title": candidate["origin_title"],
             "voice_id": candidate["voice_id"], "voice_asset": voices[candidate["voice_id"]]["asset_hash"],
             "words": candidate["words"], "length_class": candidate["length_class"],
             "paragraph": "paragraph_turns" in candidate["chunk"], "chunk": candidate["chunk"]}
    entry["seed_requested"] = int(digest({"seed": seed, "case": entry["id"]})[:8], 16)
    return entry


def _require_unique(cases):
    ids = [entry["id"] for entry in cases]
    chunks = [entry["chunk"]["chunk_id"] for entry in cases]
    if any(not _CASE_ID.fullmatch(value) for value in ids) or len(set(ids)) != len(ids) \
            or len(set(chunks)) != len(chunks):
        raise ValueError("Benchmark cases must be unique clips; refusing a duplicate corpus")


def corpus_body(bundle, config, cases, warmup, packed_voices, *, seed, sample):
    """Bind grouping, origin mappings, voices, and settings under one input hash."""
    render = {key: deepcopy(config["render"][key]) for key in SAFE_RENDER_KEYS if key in config["render"]}
    core = {"source_fingerprint": bundle["manifest"]["source_fingerprint"],
            "bundle_hash": bundle["bundle_hash"], "selection_seed": seed, "sample": sample,
            "requested_cases": len(cases), "render": render,
            "voices": packed_voices, "cases": cases, "warmup": warmup}
    provisional = digest(core)
    origins = {str(group): [{"origin": entry["origin"], "origin_title": entry["origin_title"],
                             "chunk_id": entry["chunk"]["chunk_id"]}
                            for entry in cases if entry["group"] == group]
               for group in range(GROUP_COUNT)}
    body = dict(core, format=CORPUS_FORMAT, version=1,
                book={"id": f"chapter-pipeline-benchmark-{provisional[:16]}",
                      "title": "Chapter pipeline benchmark (synthetic identity)", "authors": []},
                total_source_words=sum(entry["words"] for entry in cases),
                origin_chapters=origins)
    body["input_hash"] = digest(body)
    return body


def pack_voices(voices, output, ffmpeg):
    """Lossless FLAC transfer copies with portable relative corpus paths."""
    from epub_tts_narrator.renderer.chapter_client import _voice_assets
    bindings, assets = _voice_assets(voices, output, ffmpeg)
    packed = {}
    for voice_id, voice in sorted(voices.items()):
        transfer = assets[bindings[voice_id]]
        packed[voice_id] = {"asset_hash": bindings[voice_id],
                            "reference_audio": transfer["path"].relative_to(output).as_posix(),
                            "reference_sha256": transfer["manifest"]["audio_sha256"],
                            "reference_text": voice["reference_text"]}
    return packed


def write_corpus(body, output):
    """Persist the private corpus: 0600 files, 0700 directories, no world reads."""
    root = Path(output)
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)
    path = root / "corpus.json"
    write_json(path, dict(body, corpus_sha256=digest({k: v for k, v in body.items()})))
    path.chmod(0o600)
    return path


def load_corpus(path):
    """Verify checksums, reference bytes, traversal safety, balance, input hash."""
    path = regular_file(Path(path).resolve())
    corpus = read_json(path)
    body = {key: value for key, value in corpus.items() if key != "corpus_sha256"}
    if (corpus.get("format") != CORPUS_FORMAT or corpus.get("version") != 1
            or corpus.get("corpus_sha256") != digest(body)):
        raise ValueError("Benchmark corpus checksum mismatch")
    voices = {}
    for voice_id, record in body["voices"].items():
        relative = Path(record["reference_audio"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Corpus references must be portable relative paths")
        try:
            audio = regular_file(path.parent / relative)
        except OSError as exc:
            raise ValueError(f"Corpus reference is missing: {voice_id}") from exc
        if file_hash(audio) != record["reference_sha256"]:
            raise ValueError("Benchmark reference changed")
        voices[voice_id] = dict(record, reference_audio=str(audio.resolve()))
    _verify_body(body, voices)
    return corpus, voices


def _verify_body(body, voices):
    if body.get("input_hash") != digest({key: value for key, value in body.items() if key != "input_hash"}):
        raise ValueError("Frozen corpus input hash mismatch; the corpus is not the prepared workload")
    groups = {}
    for entry in body["cases"]:
        bound = voices.get(entry.get("voice_id")) if isinstance(entry, dict) else None
        if (not isinstance(entry, dict) or not _CASE_ID.fullmatch(str(entry.get("id")))
                or type(entry.get("group")) is not int or not 0 <= entry["group"] < GROUP_COUNT
                or bound is None or entry.get("voice_asset") != bound.get("asset_hash")
                or type(entry.get("seed_requested")) is not int or not 0 <= entry["seed_requested"] <= 0xFFFFFFFF
                or not isinstance(entry.get("chunk"), dict)
                or not isinstance(entry["chunk"].get("chunk_id"), str)
                or entry.get("words") != canonical_words(entry["chunk"].get("spoken_text", ""))
                or entry.get("length_class") != length_class(entry["chunk"].get("spoken_text", ""))):
            raise ValueError("Invalid frozen benchmark case")
        groups.setdefault(entry["group"], []).append(entry["id"])
    if (len(body["cases"]) != body.get("requested_cases") or sorted(groups) != list(range(GROUP_COUNT))
            or len({len(members) for members in groups.values()}) != 1):
        raise ValueError("Frozen corpus groups are not balanced across four benchmark chapters")
    _require_unique(body["cases"])
    warm_chunks = {entry["chunk"]["chunk_id"] for entry in body.get("warmup", [])}
    if not warm_chunks or warm_chunks & {entry["chunk"]["chunk_id"] for entry in body["cases"]}:
        raise ValueError("Frozen warmup clips must exist and stay outside the measured groups")
    if body.get("total_source_words") != sum(entry["words"] for entry in body["cases"]):
        raise ValueError("Frozen canonical source word counts changed")


def freeze_bundle(bundle_root, config_path, chapter_numbers, total, output, *, seed, sample):
    """Prepare-side driver used by the CLI; returns (corpus.json path, body)."""
    from epub_tts_narrator.renderer.book_context import resolve_book_config, voice_store
    from epub_tts_narrator.renderer.bundle import load_bundle
    from epub_tts_narrator.renderer.config import load_config
    from epub_tts_narrator.renderer.workflow import make_render_plan

    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Benchmark corpus output must be a new directory")
    bundle = load_bundle(bundle_root)
    config = resolve_book_config(bundle, load_config(config_path), work_dir=output / "unused-plan-work")
    plan = make_render_plan(bundle, config)
    if plan["errors"]:
        raise ValueError("Book render plan has unresolved errors")
    store = voice_store(bundle, config)
    voices = {key: store.load(key) for key in plan["voice_ids"]}
    if any(value is None for value in voices.values()):
        raise ValueError("All planned book voices must exist before preparing this benchmark corpus")
    cases, warmup = select_cases(plan["chapters"], voices, chapter_numbers, total=total, seed=seed)
    output.mkdir(parents=True)
    used = {entry["voice_id"] for entry in cases} | {entry["voice_id"] for entry in warmup}
    packed = pack_voices({key: voices[key] for key in sorted(used)}, output, config["render"]["ffmpeg"])
    for entry in [*cases, *warmup]:
        # Cases bind the packed transfer identity that run-time voice
        # registration and voice_bindings will use for the same reference.
        entry["voice_asset"] = packed[entry["voice_id"]]["asset_hash"]
    body = corpus_body(bundle, config, cases, warmup, packed, seed=seed, sample=sample)
    path = write_corpus(body, output)
    return path, body
