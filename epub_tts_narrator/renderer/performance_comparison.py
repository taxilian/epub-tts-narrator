"""Isolated, resumable chapter listening comparisons with frozen reference voices.

Neither half is allowed to borrow unproven audio from an earlier publication.
Both use the ordinary verified chapter renderer and the same selected voices,
sampling settings, and seed. Only the expressive performance plan differs.
"""

from __future__ import annotations

from copy import deepcopy
import html
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import monotonic

from .book_context import resolve_book_config, voice_store
from .bundle import load_bundle, make_plan
from .common import digest, file_hash, read_json, regular_file, write_json
from .diagnostics import configured_secrets, exception_detail, safe_error_text
from .run_locks import acquire_directory_lock
from .voices import VoiceStore


FORMAT = "epub-tts-narrator-performance-comparison"
VERSION = 1
MODES = ("legacy", "expressive")


def _directory(path):
    path = Path(path).expanduser().absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Comparison paths must not use symlinks")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _copy(source, destination, expected=None):
    source, destination = regular_file(source), Path(destination)
    expected = expected or file_hash(source)
    _directory(destination.parent)
    if destination.exists() or destination.is_symlink():
        if file_hash(regular_file(destination)) != expected:
            raise ValueError("Saved comparison input or artifact changed; use a new comparison directory")
        return expected
    fd, temporary = tempfile.mkstemp(prefix=".copy-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        if file_hash(temporary) != expected:
            raise ValueError("Comparison input changed while it was being copied")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return expected


def _safe_value(value, secrets):
    def safe(value):
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()
                    if not key.startswith("_") and key.lower() not in {
                        "api_key", "apikey", "password", "passwd", "token", "secret", "authorization",
                        "access_key_id", "secret_access_key", "client_secret", "access_token", "refresh_token"
                    } and not key.lower().endswith(("_api_key", "_password", "_token", "_secret"))}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, (str, Path)):
            return safe_error_text(value, secrets=secrets)
        if value is None or type(value) in (bool, int, float):
            return value
        raise ValueError("Comparison settings must contain serializable values")
    return safe(value)


def _safe_settings(config):
    """Record actual comparison settings, never credentials or unrelated services."""
    secrets = configured_secrets(config)

    result = {key: _safe_value(config.get(key, {}), secrets) for key in
              ("engine", "verification", "quality", "jev", "provider", "render", "render_pool")}
    for key in ("work_dir", "output_dir", "epub", "performance_plan", "performance_mode"):
        result["render"].pop(key, None)
    from .censorship import policy_identity
    policy = policy_identity(config.get("censorship"))
    if policy is not None:
        result["censorship"] = policy
    return result


def _read_state(root):
    path = root / "comparison.json"
    if not path.exists() and not path.is_symlink():
        return None
    value = read_json(regular_file(path))
    if (not isinstance(value, dict) or value.get("format") != FORMAT
            or value.get("version") != VERSION
            or value.get("record_sha256") != digest({key: item for key, item in value.items()
                                                     if key != "record_sha256"})):
        raise ValueError("Comparison manifest failed integrity checks")
    return value


def _save_state(root, state):
    state.pop("record_sha256", None)
    write_json(root / "comparison.json", dict(state, record_sha256=digest(state)))


def _snapshot_manifest(manifest):
    result = deepcopy(manifest)
    # The snapshot is a standalone render bundle, not a new analysis or series
    # continuation. These location links must not reach into the live books.
    result.pop("continuation", None)
    result.pop("analysis_directory", None)
    return result


def _selected_voices(bundle, config, plan):
    store = voice_store(bundle, config)
    selected = {}
    for voice_id in plan["voice_ids"]:
        binding = store.load(voice_id)
        if binding is None:
            raise ValueError(f"Comparison requires an existing selected voice: {voice_id}; cast it first")
        selected[voice_id] = binding
    return selected


class _FrozenVoiceStore(VoiceStore):
    def load(self, voice_id):
        saved = super().load(voice_id)
        if saved is None:
            raise ValueError("Frozen comparison voice is missing; comparisons never create voices")
        return saved


def _frozen_store(root, series_id, selected):
    aliases = {voice_id: item["voice_id"] for voice_id, item in selected.items()
               if voice_id != item["voice_id"]}
    store = _FrozenVoiceStore(root / "snapshot" / "voices", series_id, aliases=aliases)
    for voice_id, item in selected.items():
        saved = store.load(voice_id)
        if saved is None or saved["asset_hash"] != item["asset_hash"]:
            raise ValueError("Frozen comparison voice is missing or changed; comparison cannot create voices")
    return store


def prepare_comparison(bundle, config, chapter_number, directory):
    """Freeze source and existing voices; no model loading or voice generation.

    Returns ``bundle``, ``config``, ``voice_store``, ``directory``, and
    ``manifest``. Prepare an expressive plan against this returned bundle.
    The original book is only read. Reusing a directory with changed source,
    selected voices, or settings is an error rather than a mixed experiment.
    """
    if type(chapter_number) is not int or not 1 <= chapter_number <= len(bundle["chapters"]):
        raise ValueError("Comparison requires exactly one valid chapter number")
    config = resolve_book_config(bundle, config)
    config["render"].update(chapter_numbers=[chapter_number], performance_mode="legacy", performance_plan=None)
    plan = make_plan(bundle, config)
    if plan["errors"]:
        raise ValueError("Cannot compare an invalid chapter: " + "; ".join(plan["errors"][:3]))
    if not plan["voice_ids"]:
        raise ValueError("Comparison chapter has no spoken text")
    selected = _selected_voices(bundle, config, plan)
    root = Path(directory).expanduser().absolute()
    original = Path(bundle["directory"]).resolve()
    protected = [original, *[Path(path).resolve() for path in config.get("_voice_parents", [])],
                 Path(config["voices"]["directory"]).resolve(),
                 Path(config["render"]["work_dir"]).resolve(),
                 Path(config["render"]["output_dir"]).resolve()]
    if any(root == path or root in path.parents or path in root.parents for path in protected):
        raise ValueError("Comparison directory must be separate from source books, voices, and existing render output")
    root = _directory(root)
    lock = acquire_directory_lock(root / ".comparison.lock")
    try:
        voices = {voice_id: {key: item[key] for key in
                            ("voice_id", "candidate_id", "asset_hash", "reference_sha256")}
                  for voice_id, item in selected.items()}
        inputs = {"bundle_hash": bundle["bundle_hash"], "source_fingerprint": plan["source_fingerprint"],
                  "chapter_number": chapter_number, "series_id": plan["series_id"],
                  "voices": voices, "settings": _safe_settings(config)}
        state = _read_state(root)
        if state is not None and state.get("input_sha256") != digest(inputs):
            raise ValueError("Comparison inputs changed (source, voices, chapter, or settings); use a new comparison directory")
        if state is None:
            state = {"format": FORMAT, "version": VERSION, "input_sha256": digest(inputs),
                     "inputs": inputs, "status": "preparing", "book": deepcopy(plan["book"]),
                     "title": plan["chapters"][0]["title"], "modes": {}, "snapshot_files": {}}
            _save_state(root, state)
        snapshot = _directory(root / "snapshot")
        inventory = {}
        for name, checksum in bundle["manifest"]["sha256"].items():
            inventory["snapshot/" + name] = _copy(original / name, snapshot / name, checksum)
        manifest = _snapshot_manifest(bundle["manifest"])
        manifest_path = snapshot / "manifest.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            if read_json(regular_file(manifest_path)) != manifest:
                raise ValueError("Frozen comparison bundle manifest changed")
        else:
            write_json(manifest_path, manifest)
        inventory["snapshot/manifest.json"] = file_hash(manifest_path)
        store = VoiceStore(snapshot / "voices", config["voices"]["series_id"])
        copied = set()
        for item in selected.values():
            key = (item["voice_id"], item["candidate_id"])
            if key in copied:
                continue
            copied.add(key)
            source = Path(item["reference_audio"]).parent
            target = store._candidate_folder(*key)
            for name in ("reference.wav", "voice.json", "voice.sha256"):
                destination = target / name
                inventory[str(destination.relative_to(root))] = _copy(source / name, destination)
            if item["candidate_id"] != VoiceStore.LEGACY_CANDIDATE:
                with_lock = store.lock()
                try:
                    store.select_candidate(*key)
                finally:
                    with_lock.rmdir()
                pointer = store._selection_path(item["voice_id"])
                inventory[str(pointer.relative_to(root))] = file_hash(pointer)
        # Recheck live pointers so a recast during preparation cannot produce
        # an experiment with a partly old and partly new set of references.
        latest = _selected_voices(bundle, config, plan)
        if any(latest[key]["asset_hash"] != value["asset_hash"] for key, value in selected.items()):
            raise ValueError("Selected voices changed during comparison preparation; retry in a new directory")
        if digest(read_json(regular_file(original / "manifest.json"))) != bundle["bundle_hash"]:
            raise ValueError("Book analysis changed during comparison preparation")
        if state["snapshot_files"] and inventory != state["snapshot_files"]:
            raise ValueError("Frozen comparison input inventory changed")
        state["snapshot_files"] = inventory
        if state["status"] == "preparing":
            state["status"] = "prepared"
        _save_state(root, state)
        frozen = load_bundle(snapshot)
        frozen_config = deepcopy(config)
        frozen_config.update(_book_directory=str(snapshot), _book_local_voices=True, _voice_parents=[])
        frozen_config["voices"]["directory"] = str(snapshot / "voices")
        frozen_config["render"].update(work_dir=str(root / "legacy" / "work"),
                                       output_dir=str(root / "legacy" / "output"), epub=None)
        return {"bundle": frozen, "config": frozen_config, "voice_store": _frozen_store(root, plan["series_id"], voices),
                "directory": root, "manifest": state}
    finally:
        lock.rmdir()


def _validate_snapshot(root, state):
    for name, checksum in state["snapshot_files"].items():
        if file_hash(regular_file(root / name)) != checksum:
            raise ValueError("Frozen comparison input changed")


def _completed_mode(root, state):
    if state.get("status") != "complete":
        return False
    for name, checksum in state.get("files", {}).items():
        path = root / name
        if not path.exists():
            return False
        if file_hash(regular_file(path)) != checksum:
            raise ValueError("Saved comparison audio or metadata changed; use a new comparison directory")
    return bool(state.get("files"))


def _artifacts(root, mode, result, elapsed, config):
    from .chapter_cache import load_chapter
    chapters = result.get("chapters", [])
    if (result.get("status") != "chapters_complete" or len(chapters) != 1
            or chapters[0]["number"] != config["render"]["chapter_numbers"][0]):
        raise ValueError("Comparison renderer must return exactly the requested chapter")
    item = chapters[0]
    composed = load_chapter(item["cache_directory"], ffprobe=config["render"]["ffprobe"])
    if not item.get("audio_file") or item["audio_file"] != composed["encoded_path"]:
        raise ValueError("Comparison renderer returned no verified chapter audio")
    output = _directory(root / mode / "output")
    name = "chapter" + Path(item["audio_file"]).suffix
    files = {str((output / name).relative_to(root)): _copy(item["audio_file"], output / name,
                                                        composed["encoded_sha256"])}
    sync = {"format": "audiobook-chapter-comparison-sync", "version": 1,
            "audio_file": name, "audio_sha256": composed["encoded_sha256"],
            "sample_rate": composed["sample_rate"], "frames": composed["frames"],
            "segments": composed["segments"]}
    write_json(output / "sync.json", sync)
    files[str((output / "sync.json").relative_to(root))] = file_hash(output / "sync.json")
    duration = composed["frames"] / composed["sample_rate"]
    return {"status": "complete", "audio": str((output / name).relative_to(root)),
            "sync": str((output / "sync.json").relative_to(root)), "files": files,
            "elapsed_seconds": elapsed, "duration_seconds": duration,
            "realtime_factor": elapsed / duration if duration else None,
            "segment_count": len(composed["segments"]), "normalization": composed.get("normalization"),
            "cache_directory": str(Path(item["cache_directory"]).relative_to(root))}


def _html(root, state, performance_plan):
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    timings = {}
    for mode in MODES:
        item = state["modes"].get(mode, {})
        label = "Legacy generation" if mode == "legacy" else "Expressive performance plan"
        audio = (f'<audio id="{mode}" controls preload="metadata" src="{esc(item["audio"])}"></audio>'
                 f'<p><a href="{esc(item["audio"])}">Download chapter audio</a> · '
                 f'<a href="{esc(item["sync"])}">Source synchronization</a></p>'
                 if item.get("status") == "complete" else "<p>Audio is not ready.</p>")
        stats = (f'<p>{item["duration_seconds"]:.1f}s audio · {item["elapsed_seconds"]:.1f}s elapsed '
                 f'· {item["segment_count"]} rendered segments</p>' if item.get("status") == "complete" else "")
        cards.append(f'<section><h2>{label}</h2>{audio}{stats}<p>Status: {esc(item.get("status", "pending"))}</p></section>')
        timings[mode] = {}
        if item.get("status") == "complete":
            sync = read_json(regular_file(root / item["sync"]))
            for segment in sync["segments"]:
                identifiers = [segment.get("segment_id"), *[source.get("segment_id")
                                for source in segment.get("source_segments", [])]]
                for identifier in identifiers:
                    if identifier is not None:
                        timings[mode].setdefault(identifier, segment["chapter_start_frame"] / sync["sample_rate"])
    changed = []
    for identifier, performance in performance_plan.get("segments", {}).items():
        for control in performance.get("controls", []):
            description = " ".join(str(control[key]) for key in ("kind", "tag", "direction", "sound", "phonemes", "duration_ms")
                                   if control.get(key) not in (None, ""))
            links = []
            for mode in MODES:
                if identifier in timings[mode]:
                    second = timings[mode][identifier]
                    links.append(f'<button type="button" data-player="{mode}" data-time="{second:.6f}">'
                                 f'{mode.title()} {int(second // 60)}:{int(second % 60):02d}</button>')
            changed.append(f'<tr><td>{esc(identifier)}</td><td>{esc(control.get("text", "Event at offset " + str(control.get("offset", ""))))}</td>'
                           f'<td>{esc(description)}</td><td>{" ".join(links)}</td></tr>')
    changed_table = ('<h2>Changed controls</h2><p>Buttons start the containing passage in each version. '
                     'A paragraph containing several source segments has one shared playback start.</p>'
                     '<table><thead><tr><th>Source segment</th><th>Words</th><th>Control</th><th>Listen</th></tr></thead><tbody>'
                     + "".join(changed) + '</tbody></table>' if changed else '<p>No additional scoped controls are present in this plan.</p>')
    controls = json.dumps(performance_plan, ensure_ascii=False, indent=2)
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                '<title>Chapter listening comparison</title><style>body{font:17px/1.55 system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;'
                'background:#f5f4f0;color:#17212b}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.5rem}'
                'section{background:white;border:1px solid #d5d8dc;border-radius:12px;padding:1.5rem}audio{width:100%}pre{white-space:pre-wrap;'
                'overflow-wrap:anywhere;background:#fff;padding:1rem}a{color:#155b8a}table{width:100%;border-collapse:collapse}'
                'td,th{text-align:left;vertical-align:top;padding:.6rem;border-bottom:1px solid #c9cdd1;overflow-wrap:anywhere}'
                'button{padding:.45rem;margin:.2rem;border:1px solid #9eabb5;border-radius:5px;cursor:pointer}</style>'
                f'<h1>{esc(state["book"]["title"])}</h1><p>Chapter {state["inputs"]["chapter_number"]}: {esc(state["title"])}</p>'
                '<p>Both versions use the same frozen source, selected reference voices, sampling settings, and base seed. '
                'Individual take seeds also depend on request identity, so the two methods do not share identical sampling noise. '
                'Generation is stochastic; this comparison measures one rendered chapter per method, not a guarantee of quality.</p>'
                f'<main>{"".join(cards)}</main><p>Overall status: {esc(state["status"])}</p>'
                '<p>Elapsed time includes generation, verification, retries, and assembly. Resumed runs can include cache reuse. '
                'Source timing is independently measured for each version; identical playback positions need not identify identical words.</p>'
                '<p><a href="comparison.json">Comparison bindings and statistics</a> · '
                '<a href="performance-plan.json">Expressive controls</a></p>'
                + changed_table + f'<details><summary>Full performance plan</summary><pre>{esc(controls)}</pre></details>'
                '<script>document.querySelectorAll("button[data-player]").forEach(button=>button.addEventListener("click",()=>{'
                'document.querySelectorAll("audio").forEach(audio=>audio.pause());const audio=document.getElementById(button.dataset.player);'
                'audio.currentTime=Number(button.dataset.time);audio.play().catch(()=>{});}));</script></html>')
    from .common import write_private
    write_private(root / "index.html", document)


def compare_chapter(bundle, config, engine, verifier, *, chapter_number, directory,
                    performance_plan, notify=print, render_method=None):
    """Render both methods with one engine instance; rerun to resume either half.

    ``performance_plan`` is a validated plan dictionary or its saved JSON path.
    Use :func:`prepare_comparison` before planning so its bundle binding matches
    this isolated snapshot. The renderer accepts ``voice_store_override`` to
    keep alias lookup and every potential write inside the snapshot.
    """
    prepared = prepare_comparison(bundle, config, chapter_number, directory)
    root, state = prepared["directory"], prepared["manifest"]
    plan = (read_json(regular_file(performance_plan)) if isinstance(performance_plan, (str, Path))
            else deepcopy(performance_plan))
    if not isinstance(plan, dict):
        raise ValueError("Comparison requires a validated expressive performance plan")
    from ..performance_planning import _validate_state
    rows = prepared["bundle"]["chapters"][chapter_number - 1]["rows"]
    _validate_state(plan, prepared["bundle"], [chapter_number], rows, complete=True)
    if render_method is None:
        from .workflow import render
        render_method = render
    lock = acquire_directory_lock(root / ".comparison.lock")
    try:
        state = _read_state(root)
        _validate_snapshot(root, state)
        runtime = _safe_value({"engine": engine.identity, "verifier": verifier.identity}, configured_secrets(config))
        if state.get("runtime") is not None and state["runtime"] != runtime:
            raise ValueError("Comparison engine or verifier identity changed; use a new comparison directory")
        if state.get("performance_sha256") is not None and state["performance_sha256"] != digest(plan):
            raise ValueError("Expressive comparison plan changed; use a new comparison directory")
        state.update(runtime=deepcopy(runtime), performance_sha256=digest(plan))
        _save_state(root, state)
        plan_path = root / "performance-plan.json"
        if plan_path.exists() or plan_path.is_symlink():
            if read_json(regular_file(plan_path)) != plan:
                raise ValueError("Saved expressive comparison plan changed")
        else:
            write_json(plan_path, plan)
        for mode in MODES:
            if _completed_mode(root, state["modes"].get(mode, {})):
                notify(f"Reusing completed {mode} chapter comparison")
                continue
            _validate_snapshot(root, state)
            local = deepcopy(prepared["config"])
            local["render"].update(work_dir=str(root / mode / "work"), output_dir=str(root / mode / "output"),
                                   performance_mode=mode, performance_plan=str(plan_path) if mode == "expressive" else None)
            started = monotonic()
            previous = state["modes"].get(mode, {})
            elapsed = previous.get("elapsed_seconds", 0)
            state["status"] = "rendering"
            state["modes"][mode] = {"status": "rendering", "elapsed_seconds": elapsed}
            _save_state(root, state)
            _html(root, state, plan)
            notify(f"Rendering {mode} comparison, chapter {chapter_number}")
            try:
                store = _frozen_store(root, state["inputs"]["series_id"], state["inputs"]["voices"])
                result = render_method(prepared["bundle"], local, engine, verifier, notify=notify,
                                       chapters_only=True, voice_store_override=store)
                state["modes"][mode] = _artifacts(root, mode, result, elapsed + monotonic() - started, local)
                _validate_snapshot(root, state)
            except BaseException as exc:
                state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                state["modes"][mode] = {"status": state["status"], "elapsed_seconds": elapsed + monotonic() - started,
                                        "error": exception_detail(exc)}
                _save_state(root, state)
                _html(root, state, plan)
                raise
            _save_state(root, state)
            _html(root, state, plan)
        state["status"] = "complete"
        _save_state(root, state)
        _html(root, state, plan)
        notify(f"Chapter comparison ready: {root / 'index.html'}")
        return {"status": "complete", "directory": str(root), "html": str(root / "index.html"),
                "manifest": str(root / "comparison.json"), "modes": deepcopy(state["modes"])}
    finally:
        lock.rmdir()
