#!/usr/bin/env python3
"""Listen to Drama 3 cloning a voice from one reference clip against three.

``run`` works per chosen voice. It never changes the book, its voices or the
casting selections:

1. Picks separate book excerpts the character speaks (the casting audition
   rules, with no excerpt in two clips) plus a few other lines of theirs to test.
2. Has one hosted Gemini voice, made from the voice's saved description, speak
   every excerpt. Each clip is verified like a casting reference. The hosted
   voice is deleted once all clips are verified, or when one fails.
3. Renders the test lines and one shared sentence with local Drama 3 under three
   conditions: ``saved`` (the book's selected reference), ``one`` (new clip 1)
   and ``all`` (every new clip, one speaker). There is no chapter history, and
   each line uses one seed in every condition, so only the references differ.
4. Writes ``index.html`` beside the audio for side-by-side listening.

``score`` optionally adds speaker-verification similarity from a local WavLM
x-vector model directory (for example ``microsoft/wavlm-base-plus-sv``,
downloaded separately; nothing is fetched). Higher similarity to a voice's own
designer audio means a closer clone. Lower similarity between different
voices' renders of the shared sentence means they are easier to tell apart.

Rerun the same command to resume. Verified clips and finished renders are
reused, and a hosted voice from an interrupted run is reused. The run makes
paid Gemini calls and briefly stores hosted voices in your Google project.
It needs a local ``drama3_mlx`` engine and never uses Runpod or remote workers.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import html
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epub_tts_narrator.renderer.audio_naming import descriptive_audio_filename
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.samples import _WORDS, _candidate_data, _choose


VERSION = 1
CONDITIONS = ("saved", "one", "all")
CONDITION_LABELS = {"saved": "Saved reference (1 clip)", "one": "New clip 1 only", "all": "All new clips"}
SHARED_TEXT = ("I told you before, and I'll tell you again: we leave at first light, "
               "whether the others are ready or not.")
LINE_MIN_WORDS, LINE_MAX_WORDS, LINE_TARGET_WORDS = 6, 30, 14
RENDER_ATTEMPTS = 3


# ---------------------------------------------------------------- selection

def resolve_voice(bundle, selector):
    """A cast voice ID, a character name or alias (case-insensitive), or ``narrator``."""
    cast = bundle["cast"]
    voice_ids = {voice["id"] for voice in cast["voices"]}
    if selector in voice_ids:
        return selector
    wanted = selector.casefold()
    if wanted == "narrator":
        narrators = [role["voice_id"] for role in cast["roles"] if role["id"] == "narrator_main"]
        if narrators:
            return narrators[0]
    exact = {item["voice_id"] for item in cast.get("characters", [])
             if str(item.get("name", "")).casefold() == wanted}
    if not exact:
        exact = {item["voice_id"] for item in cast.get("characters", [])
                 if any(str(alias).casefold() == wanted for alias in item.get("aliases", []))}
    if len(exact) == 1:
        return next(iter(exact))
    if exact:
        raise ValueError(f"{selector!r} names more than one voice; give a voice ID instead")
    raise ValueError(f"No cast voice, character name or alias matches {selector!r}")


def voice_name(bundle, voice_id):
    for item in bundle["cast"].get("characters", []):
        if item.get("voice_id") == voice_id and item.get("name"):
            return str(item["name"])
    if any(role["voice_id"] == voice_id and role["id"] == "narrator_main" for role in bundle["cast"]["roles"]):
        return "Narrator"
    return voice_id


def disjoint_scripts(bundle, voice_id, canonical_id, count):
    """Up to ``count`` casting auditions in which no book segment appears twice."""
    source_base, _rejected, candidates, _canonical = _candidate_data(bundle, voice_id, canonical_id)
    scripts, used = [], set()
    for _ in range(count):
        remaining = [item for item in candidates if item["source"]["segment_id"] not in used]
        chosen = _choose(remaining, source_base)
        if chosen is None:
            break
        scripts.append(chosen)
        used.update(chosen["source"]["segment_ids"])
    return scripts, candidates, used


def test_lines(candidates, used_segments, count):
    """Other lines the character speaks, preferring whole sentences near 14 words, one per segment and chapter."""
    pool = {}
    for item in candidates:
        segment = item["source"]["segment_id"]
        words = item["source"]["word_count"]
        if segment in used_segments or not LINE_MIN_WORDS <= words <= LINE_MAX_WORDS:
            continue
        key = (item["score"][0], item["score"][2], abs(words - LINE_TARGET_WORDS), item["order"])
        if segment not in pool or key < pool[segment][0]:
            pool[segment] = (key, item)
    ranked = [item for _key, item in sorted(pool.values(), key=lambda pair: pair[0])]
    chosen, chapters = [], set()
    for item in ranked:
        if item["source"]["chapter_number"] not in chapters:
            chosen.append(item)
            chapters.add(item["source"]["chapter_number"])
        if len(chosen) == count:
            break
    for item in ranked:
        if len(chosen) == count:
            break
        if item not in chosen:
            chosen.append(item)
    return [{"text": item["reference_text"], "segment_id": item["source"]["segment_id"]}
            for item in sorted(chosen, key=lambda item: item["order"])]


def make_plan(bundle, voice_id, canonical_id, *, clips, lines, shared_text, description):
    scripts, candidates, used = disjoint_scripts(bundle, voice_id, canonical_id, clips)
    if len(scripts) < clips:
        raise ValueError(f"{voice_name(bundle, voice_id)} has book speech for only {len(scripts)} separate "
                         f"audition script(s), not {clips}; choose a character with more lines")
    return {"format": "reference-clip-experiment", "version": VERSION, "voice_id": voice_id,
            "name": voice_name(bundle, voice_id), "description": description,
            "source_fingerprint": bundle["manifest"]["source_fingerprint"],
            "scripts": [{"text": item["reference_text"], "source": item["source"]} for item in scripts],
            "lines": [{"text": shared_text, "shared": True},
                      *test_lines(candidates, used, lines)]}


def voice_directory(root, plan):
    slug = "".join(character if character.isalnum() else "-" for character in plan["name"]).strip("-").lower()
    return Path(root) / f"{slug or 'voice'}-{digest(plan['voice_id'])[:8]}"


# ---------------------------------------------------------------- clips

def clip_request(plan, index, *, designer, base_seed):
    return {"kind": "create_voice", "experiment": "reference-clips", "version": VERSION,
            "voice_id": plan["voice_id"], "profile": {"name": f"{plan['name']} reference clip {index}"},
            "designer": designer, "prompt": plan["description"], "text": plan["scripts"][index - 1]["text"],
            "clip": index, "base_seed": base_seed}


def hosted_generator(designer, prompt, text, hosted, label):
    """Speak ``text`` with the one hosted voice kept in ``hosted``, then move the take to ``path``.

    The Gemini designer keys its hosted voice by the take's folder, so every clip
    is first written there; ``verified_take`` then owns the moved file.
    """
    from epub_tts_narrator.renderer.voice_bindings import design_voice

    def generate(path, seed):
        hosted.mkdir(parents=True, exist_ok=True)
        staged = hosted / descriptive_audio_filename("hosted-take", label, f"seed-{seed}", suffix=".wav")
        staged.unlink(missing_ok=True)  # Our own leftover from an interrupted attempt.
        metadata = design_voice(designer, prompt, text, staged, seed=seed)
        os.replace(staged, path)
        return metadata
    return generate


def prepare_clips(session, plan, folder, notify=print):
    """Verified clips for this plan from one hosted voice; reused when already saved."""
    record_path = folder / "clips.json"
    if record_path.is_file():
        record = read_json(record_path)
        for clip in record["clips"]:
            if file_hash(clip["reference_audio"]) != clip["reference_sha256"]:
                raise ValueError(f"Saved experiment clip changed: {clip['reference_audio']}")
        return record["clips"]
    from epub_tts_narrator.renderer.voice_bindings import designer_identity
    from epub_tts_narrator.renderer.voice_prompts import audiobook_voice_prompt, designer_prompt
    from epub_tts_narrator.renderer.workflow import SpeechVerificationFailed, verified_take
    designer = session._get_designer()
    identity = designer_identity(designer)
    if identity.get("engine") != "gemini_voice_design" or not callable(getattr(designer, "release_voice", None)):
        raise ValueError("This experiment needs the Gemini voice designer: only it keeps one hosted voice "
                         "across several scripts. Set [voice_design] type = \"gemini\".")
    verifier = session._get_verifier()
    prompt = designer_prompt(audiobook_voice_prompt(plan["description"]), identity)
    hosted = folder / "hosted-voice"
    clips = []
    try:
        for index, script in enumerate(plan["scripts"], 1):
            request = clip_request(plan, index, designer=identity, base_seed=session.config["render"]["seed"])
            notify(f"  {plan['name']}: clip {index} of {len(plan['scripts'])}")
            wav, accepted = verified_take(
                script["text"], hosted_generator(designer, prompt, script["text"], hosted,
                                                 f"{plan['name']}-clip-{index}"),
                folder / f"clip-{index}", request, verifier, session.config,
                notify=notify, purpose="reference")
            clips.append({"index": index, "reference_audio": str(Path(wav).resolve()),
                          "reference_text": accepted["report"].get("reference_text", script["text"]),
                          "reference_sha256": file_hash(wav)})
        write_json(record_path, {"format": "reference-clip-experiment-clips", "version": VERSION,
                                 "plan_sha256": digest(plan), "clips": clips})
        designer.release_voice(hosted)
    except SpeechVerificationFailed:
        # A clip that exhausted its attempts cannot share a fresh hosted voice
        # with the clips already made, so this voice is deleted. An
        # interruption keeps it, and a rerun continues with the same voice.
        designer.release_voice(hosted)
        notify(f"  {plan['name']}: a clip failed verification; its hosted voice was deleted. "
               f"Remove {folder} to try a fresh voice.")
        raise
    return clips


# ---------------------------------------------------------------- renders

def reference_voice(binding):
    return {"kind": "reference", "reference_audio": binding["reference_audio"],
            "reference_text": binding["reference_text"], "reference_sha256": binding["reference_sha256"]}


def condition_references(saved, clips):
    return {"saved": [reference_voice(saved)], "one": [reference_voice(clips[0])],
            "all": [reference_voice(clip) for clip in clips]}


def render_name(plan, condition, number):
    return descriptive_audio_filename(f"{condition}-line-{number:02d}", plan["name"], suffix=".wav")


def render_line(engine, references, text, output, seed):
    """One take of ``text`` from these references as speaker 0, without chapter history."""
    from epub_tts_narrator.renderer.engines import EngineError, prepare_synthesis
    _prepared, directed, _delivery = prepare_synthesis(text)
    wire = f"<|speaker:0|> {directed}"
    model = engine._load()
    encoded = []
    for voice in references:
        reference = engine._reference(voice)
        encoded.append(replace(reference, text=f"<|speaker:0|> {reference.text}"))
    segment = engine._directing_segment(model, wire, encoded, [], seed)
    if segment is None:
        raise EngineError("Drama 3 returned no audio frames")
    engine._save_segment(segment, model, wire, Path(output))
    return {"finish_reason": segment.finish_reason, "prompt_rows": segment.prompt_rows}


def render_voice(engine, plan, folder, references, *, seed_base, sampling, notify=print):
    from epub_tts_narrator.renderer.engines import EngineError
    from epub_tts_narrator.renderer.request_sampling import request_sampling
    results = {}
    for condition in CONDITIONS:
        out = folder / "renders" / condition
        out.mkdir(parents=True, exist_ok=True)
        for number, line in enumerate(plan["lines"], 1):
            wav = out / render_name(plan, condition, number)
            record_path = wav.with_suffix(".json")
            binding = {"text": line["text"], "references": [item["reference_sha256"] for item in references[condition]],
                       "sampling": sampling, "engine": engine.identity}
            if record_path.is_file() and wav.is_file():
                record = read_json(record_path)
                if record.get("binding_sha256") == digest(binding) and file_hash(wav) == record.get("audio_sha256"):
                    results[(condition, number)] = record
                    continue
                raise ValueError(f"Saved render {wav} was made from other inputs; remove it to render again")
            notify(f"  {plan['name']}: {condition} line {number} of {len(plan['lines'])}")
            failures = []
            with request_sampling(sampling) if sampling else nullcontext():
                for attempt in range(RENDER_ATTEMPTS):
                    # The same seed per line in every condition; a retry moves all conditions' later attempts alike.
                    seed = (seed_base + number * 1000 + attempt) % 2**32
                    try:
                        metadata = render_line(engine, references[condition], line["text"], wav, seed)
                        break
                    except EngineError as exc:
                        wav.unlink(missing_ok=True)
                        failures.append(str(exc))
                else:
                    raise EngineError(f"{plan['name']} {condition} line {number} failed {RENDER_ATTEMPTS} times: "
                                      + "; ".join(failures))
            record = {"binding_sha256": digest(binding), "audio_sha256": file_hash(wav), "seed": seed,
                      "attempt": attempt + 1, **metadata}
            write_json(record_path, record)
            results[(condition, number)] = record
    return results


# ---------------------------------------------------------------- report

def relative(path, root):
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def write_index(root, voices, scores=None):
    """One page: each voice's designer audio, then every line under every condition."""
    root = Path(root)
    escape = html.escape
    parts = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
             "<title>Reference clip comparison</title><style>"
             "body{font:15px/1.4 system-ui,sans-serif;margin:16px;max-width:1200px}"
             "table{border-collapse:collapse;width:100%;margin:8px 0 24px}"
             "td,th{border:1px solid #ccc;padding:6px;vertical-align:top;text-align:left}"
             "audio{width:220px}.text{max-width:360px}.num{text-align:right}</style></head><body>",
             "<h1>Reference clip comparison</h1>",
             "<p>Each line uses one seed in every condition and no chapter history, so only the "
             "references differ. <b>saved</b> is the book's selected reference; <b>one</b> and "
             "<b>all</b> use clips from one new hosted voice.</p>"]
    if scores:
        parts.append(score_tables(scores))
    for voice in voices:
        plan, folder = voice["plan"], Path(voice["folder"])
        parts.append(f"<h2>{escape(plan['name'])}</h2><p class='text'>{escape(plan['description'])}</p>")
        parts.append("<table><tr><th>Designer audio</th><th>Script</th></tr>")
        parts.append(f"<tr><td><audio controls preload='none' src='{escape(relative(voice['saved']['reference_audio'], root))}'>"
                     f"</audio><br>saved reference</td><td class='text'>{escape(voice['saved']['reference_text'])}</td></tr>")
        for clip in voice["clips"]:
            parts.append(f"<tr><td><audio controls preload='none' src='{escape(relative(clip['reference_audio'], root))}'>"
                         f"</audio><br>new clip {clip['index']}</td><td class='text'>{escape(clip['reference_text'])}</td></tr>")
        parts.append("</table><table><tr><th>Line</th>"
                     + "".join(f"<th>{escape(CONDITION_LABELS[condition])}</th>" for condition in CONDITIONS) + "</tr>")
        for number, line in enumerate(plan["lines"], 1):
            label = "shared sentence" if line.get("shared") else f"line {number - 1}"
            cells = []
            for condition in CONDITIONS:
                wav = folder / "renders" / condition / render_name(plan, condition, number)
                cells.append(f"<td><audio controls preload='none' src='{escape(relative(wav, root))}'></audio></td>"
                             if wav.is_file() else "<td>not rendered</td>")
            parts.append(f"<tr><td class='text'><b>{escape(label)}</b><br>{escape(line['text'])}</td>{''.join(cells)}</tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    page = root / "index.html"
    page.write_text("".join(parts), encoding="utf-8")
    return page


def score_tables(scores):
    escape = html.escape
    rows = ["<h2>Speaker similarity</h2><p>Cosine similarity of WavLM speaker embeddings. "
            "<b>Clone fidelity</b>: renders against that voice's own designer audio (higher is closer). "
            "<b>Separation</b>: the shared sentence between different voices (lower is easier to tell apart); "
            "the designer row compares the designer recordings themselves.</p>",
            "<table><tr><th>Voice</th>" + "".join(f"<th>{escape(CONDITION_LABELS[c])}</th>" for c in CONDITIONS) + "</tr>"]
    for name, values in scores["fidelity"].items():
        rows.append(f"<tr><td>{escape(name)}</td>" + "".join(
            f"<td class='num'>{values[c]:.3f}</td>" for c in CONDITIONS) + "</tr>")
    rows.append("</table><table><tr><th>Mean separation</th><th>Designer audio</th>"
                + "".join(f"<th>{escape(CONDITION_LABELS[c])}</th>" for c in CONDITIONS) + "</tr>")
    separation = scores["separation"]
    rows.append("<tr><td>all voice pairs</td>" + "".join(
        f"<td class='num'>{separation[key]:.3f}</td>" for key in ("designer", *CONDITIONS)) + "</tr></table>")
    return "".join(rows)


# ---------------------------------------------------------------- scoring

def cosine(left, right):
    import numpy as np
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def mean_pairwise(vectors):
    values = [cosine(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    return sum(values) / len(values) if values else float("nan")


def centroid(vectors):
    import numpy as np
    return np.mean([np.asarray(vector) / np.linalg.norm(vector) for vector in vectors], axis=0)


def summarize_scores(voices, embed):
    """Clone fidelity per voice and condition, and cross-voice separation of the shared sentence."""
    fidelity, shared = {}, {condition: [] for condition in CONDITIONS}
    designer = []
    for voice in voices:
        plan, folder = voice["plan"], Path(voice["folder"])
        saved = embed(voice["saved"]["reference_audio"])
        new = centroid([embed(clip["reference_audio"]) for clip in voice["clips"]])
        designer.append(new)
        targets = {"saved": saved, "one": new, "all": new}
        fidelity[plan["name"]] = {}
        for condition in CONDITIONS:
            values = []
            for number, line in enumerate(plan["lines"], 1):
                vector = embed(folder / "renders" / condition / render_name(plan, condition, number))
                values.append(cosine(vector, targets[condition]))
                if line.get("shared"):
                    shared[condition].append(vector)
            fidelity[plan["name"]][condition] = sum(values) / len(values)
    separation = {"designer": mean_pairwise(designer),
                  **{condition: mean_pairwise(vectors) for condition, vectors in shared.items()}}
    return {"fidelity": fidelity, "separation": separation}


def wavlm_embedder(model_directory):
    import numpy as np
    import soundfile
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoFeatureExtractor, WavLMForXVector
    extractor = AutoFeatureExtractor.from_pretrained(model_directory, local_files_only=True)
    model = WavLMForXVector.from_pretrained(model_directory, local_files_only=True).eval()
    rate = extractor.sampling_rate
    cache = {}

    def embed(path):
        path = str(Path(path).resolve())
        if path not in cache:
            audio, source_rate = soundfile.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if source_rate != rate:
                from math import gcd
                factor = gcd(source_rate, rate)
                audio = resample_poly(audio, rate // factor, source_rate // factor).astype(np.float32)
            inputs = extractor(audio, sampling_rate=rate, return_tensors="pt")
            with torch.no_grad():
                cache[path] = model(**inputs).embeddings[0].numpy()
        return cache[path]
    return embed


# ---------------------------------------------------------------- commands

def load_voices(root):
    state = read_json(Path(root) / "experiment.json")
    voices = []
    for entry in state["voices"]:
        folder = Path(entry["folder"])
        plan = read_json(folder / "plan.json")
        clips = read_json(folder / "clips.json")["clips"]
        voices.append({"plan": plan, "folder": str(folder), "clips": clips, "saved": entry["saved"]})
    return voices


def run(args):
    from epub_tts_narrator.renderer.bundle import load_bundle
    from epub_tts_narrator.renderer.casting import CastingSession
    from epub_tts_narrator.renderer.config import load_config
    from epub_tts_narrator.renderer.role_sampling import dialogue_overrides
    bundle = load_bundle(args.book)
    config = load_config(args.config)
    if config["engine"].get("type", "drama3_mlx") != "drama3_mlx":
        raise ValueError("This experiment renders with a local drama3_mlx [engine]; remote workers are not used")
    session = CastingSession(bundle, config)
    try:
        store = session.store
        root = Path(args.output or Path(session.config["render"]["work_dir"]) / "reference-clip-experiment")
        selected = []
        for selector in args.voices:
            voice_id = resolve_voice(bundle, selector)
            saved = store.load(voice_id)
            if saved is None:
                raise ValueError(f"{selector!r} has no selected voice yet; cast it first")
            description = (saved.get("creation") or {}).get("requested_description") \
                or session._draft(voice_id, saved)["prompt"]
            plan = make_plan(bundle, voice_id, store.canonical_id, clips=args.clips, lines=args.lines,
                             shared_text=args.shared_text, description=description)
            selected.append((plan, reference_voice(saved)))
        if args.dry_run:
            for plan, _saved in selected:
                words = [len(_WORDS.findall(script["text"])) for script in plan["scripts"]]
                print(f"{plan['name']} ({plan['voice_id']}): {len(words)} clips of {words} words; "
                      f"{len(plan['lines']) - 1} test lines plus the shared sentence")
            print(f"Output: {root}")
            return 0
        root.mkdir(parents=True, exist_ok=True)
        entries = []
        for plan, saved in selected:
            folder = voice_directory(root, plan)
            folder.mkdir(parents=True, exist_ok=True)
            plan_path = folder / "plan.json"
            if plan_path.is_file() and read_json(plan_path) != plan:
                raise ValueError(f"{plan['name']}: the saved experiment plan differs from this run's "
                                 f"(book, description or options changed); remove {folder} to start it again")
            write_json(plan_path, plan)
            entries.append({"voice_id": plan["voice_id"], "folder": str(folder), "saved": saved})
        write_json(root / "experiment.json", {"format": "reference-clip-experiment-run", "version": VERSION,
                                              "book": str(Path(args.book).resolve()), "voices": entries})
        print("Making reference clips (Gemini)")
        clips = {}
        for plan, _saved in selected:
            clips[plan["voice_id"]] = prepare_clips(session, plan, voice_directory(root, plan))
        print("Rendering with local Drama 3")
        engine = session._get_engine()
        overrides = dialogue_overrides(session.config)
        narrators = {role["voice_id"] for role in bundle["cast"]["roles"] if role["kind"] == "narration"}
        for plan, saved in selected:
            sampling = {} if plan["voice_id"] in narrators else overrides
            render_voice(engine, plan, voice_directory(root, plan),
                         condition_references(saved, clips[plan["voice_id"]]),
                         seed_base=session.config["render"]["seed"], sampling=sampling)
        scores_path = root / "scores.json"
        page = write_index(root, load_voices(root), read_json(scores_path) if scores_path.is_file() else None)
        print(f"Open {page}")
        return 0
    finally:
        session.close()


def score(args):
    root = Path(args.output)
    voices = load_voices(root)
    scores = summarize_scores(voices, wavlm_embedder(args.model))
    write_json(root / "scores.json", scores)
    page = write_index(root, voices, scores)
    print(json.dumps(scores, indent=2))
    print(f"Open {page}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="make clips, render every condition and write index.html")
    run_parser.add_argument("book", help="published book directory")
    run_parser.add_argument("--voices", nargs="+", required=True,
                            help="voice IDs, character names or aliases, or 'narrator' (two or more to compare voices)")
    run_parser.add_argument("--config", help="configuration file (default: the usual lookup)")
    run_parser.add_argument("-o", "--output", help="experiment directory (default: WORK_DIR/reference-clip-experiment)")
    run_parser.add_argument("--clips", type=int, default=3, help="new reference clips per voice (default 3)")
    run_parser.add_argument("--lines", type=int, default=3, help="test lines per voice besides the shared sentence")
    run_parser.add_argument("--shared-text", default=SHARED_TEXT, help="sentence every voice speaks")
    run_parser.add_argument("--dry-run", action="store_true", help="show the chosen clips and lines; no models or API calls")
    score_parser = commands.add_parser("score", help="add speaker-similarity scores to a finished run")
    score_parser.add_argument("output", help="experiment directory from run")
    score_parser.add_argument("--model", required=True, help="local WavLM x-vector model directory")
    args = parser.parse_args(argv)
    if args.command == "run":
        if not 2 <= args.clips <= 5:
            parser.error("--clips must be from 2 through 5")
        if not 0 <= args.lines <= 8:
            parser.error("--lines must be from 0 through 8")
        return run(args)
    return score(args)


if __name__ == "__main__":
    sys.exit(main())
