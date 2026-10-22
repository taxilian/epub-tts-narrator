"""Verified audiobook rendering with local and remote speech workers."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import sys

from epub_tts_narrator.provider import ProviderError

from .bundle import load_bundle, make_plan
from .config import load_config, select_voice_designer
from .book_context import resolve_book_config, voice_store


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("plan", "render", "assemble", "cleanup", "voices", "create-voices", "import-voice", "cast", "voice-aliases"):
        command = commands.add_parser(name)
        command.add_argument("bundle", type=Path)
        command.add_argument("--config", type=Path, help="Global TOML file (default: ./epub-tts-narrator.toml; legacy fallback: ./epub-audiobook.toml)")
        command.add_argument("--work-dir", type=Path, help="Book-specific cache directory (default: BOOK/render-work)")
        command.add_argument("-o", "--output", type=Path, help="Book-specific audio output (default: BOOK/audiobook)")
        command.add_argument("--epub", type=Path, help="Original EPUB for audiobook cover extraction")
        if name in {"plan", "render", "assemble"}:
            command.add_argument("--chapters", help="Manifest chapter numbers, e.g. 1-3,5 (default: all)")
            command.add_argument("--allow-review", action="store_true", default=None,
                                 help="Compatibility option; review notes are advisory and need no override")
        if name in {"render", "assemble"}:
            command.add_argument("--chapter-format", choices=("m4a", "mp3"),
                                 help="Individual chapter format (default: configured format, normally m4a)")
        if name == "render":
            command.add_argument("--chapters-only", action="store_true",
                                 help="Generate and save chapter audio; assemble the M4B later")
        if name == "cleanup":
            command.add_argument("--dry-run", action="store_true",
                                 help="List removable files and space savings without deleting anything")
            command.add_argument("--discard-render-cache", action="store_true",
                                 help="Also delete accepted speech and chapter caches; future rendering regenerates speech. Never deletes voice samples or finished audiobooks")
        if name == "voice-aliases":
            command.add_argument("--set", dest="alias_assignments", nargs=2, action="append", default=[],
                                 metavar=("SOURCE", "TARGET"), help="Share TARGET's voice for SOURCE in this book and descendants")
            command.add_argument("--remove", dest="alias_removals", action="append", default=[], metavar="SOURCE",
                                 help="Remove a local mapping, restoring any inherited or automatic mapping")
        if name == "import-voice":
            command.add_argument("--voice-id", required=True)
            command.add_argument("--audio", type=Path, required=True)
            command.add_argument("--transcript", type=Path, required=True,
                                 help="UTF-8 file containing the exact reference recording transcript")
        if name == "cast":
            command.add_argument("--voice-designer", choices=("local", "gemini", "elevenlabs"),
                                 help="Override voice creation (default: Gemini when configured, otherwise local)")
            command.add_argument("--action", default="interactive",
                                 choices=("interactive", "list", "prepare-samples", "refresh-prompts", "normalize-samples", "generate", "generate-missing", "regenerate-all", "edit", "play", "accept", "recover"),
                                 help="Default: interactive voice casting; refresh-prompts rewrites draft descriptions and scripts using the analysis model")
            command.add_argument("--include-inherited", action="store_true",
                                 help="Also replace inherited voices with book-local overrides (--action regenerate-all)")
            command.add_argument("--voice-id")
            command.add_argument("--candidate-id")
            command.add_argument("--prompt-file", type=Path, help="UTF-8 voice description for the next generation")
            command.add_argument("--text-file", type=Path, help="UTF-8 spoken sample text for the next generation")
            command.add_argument("--attempt-file", type=Path,
                                 help="Saved attempt JSON to recheck without resynthesis (--action recover)")
            command.add_argument("--keep-unselected", action="store_true",
                                 help="Save a recovered take without changing the active voice (--action recover)")
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--config", type=Path, help="Global TOML file (default: ./epub-tts-narrator.toml; legacy fallback: ./epub-audiobook.toml)")
    serve = commands.add_parser("serve", help="Serve local Drama 3 through a Fish-compatible /v1/tts endpoint")
    serve.add_argument("--config", type=Path, help="Global TOML file containing [engine] and [server] settings")
    chapters = commands.add_parser("serve-chapters", help="Serve resumable chapter jobs with compressed audio artifacts")
    chapters.add_argument("--config", type=Path, help="Global TOML file containing [chapter_server] and renderer settings")
    return root


def chapter_numbers(value, maximum):
    """Expand bounded manifest ranges in source order, ignoring duplicate entries."""
    selected = set()
    for part in value.split(","):
        match = re.fullmatch(r"([0-9]+)(?:\s*-\s*([0-9]+))?", part.strip())
        if match is None:
            raise ValueError("--chapters must contain chapter numbers or ranges, such as 1-3,5")
        start = int(match[1])
        end = int(match[2]) if match[2] else start
        if not 1 <= start <= end <= maximum:
            raise ValueError(f"--chapters entries must be ascending ranges within 1-{maximum}")
        selected.update(range(start, end + 1))
    return sorted(selected)


def doctor(config):
    from .drama3_models import DEFAULT_DRAMA3_MODEL, get_drama3_model, supports_mlx_audio_drama3

    if config.get("remote_chapters", {}).get("url"):
        checks = {key: shutil.which(config["render"][key]) for key in ("ffmpeg", "ffprobe")}
        return {"ready": all(checks.values()), "checks": checks, "packages": {},
                "note": "Remote chapter mode needs no local synthesis or ASR model. Service connectivity is checked when rendering."}
    workers = config.get("render_pool", {}).get("workers", [])
    remote_only = bool(workers) and all(worker["type"] != "local" for worker in workers)
    faster_whisper = config["verification"].get("type", "mlx_whisper") == "faster_whisper"
    packages = ["drama3-mlx", "mlx", "mlx-whisper", "mlx-audio"]
    if faster_whisper:
        packages = ["faster-whisper", "ctranslate2", "tokenizers", "av"]
        if not remote_only and config["engine"]["type"] == "drama3_mlx":
            selected = get_drama3_model(config["engine"].get("model", DEFAULT_DRAMA3_MODEL))
            packages += ["mlx", "mlx-audio" if selected.runtime == "mlx_audio" else "drama3-mlx"]
        if config.get("voice_design", {}).get("type") == "qwen3_mlx":
            packages += ["mlx", "mlx-audio"]
    dependencies = {}
    for name in dict.fromkeys(packages):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    checkpoint = Path(config["engine"]["checkpoint"]) if config["engine"]["checkpoint"] else None
    verifier = Path(config["verification"]["model"])
    checks = {
        "ffmpeg": shutil.which(config["render"]["ffmpeg"]),
        "ffprobe": shutil.which(config["render"]["ffprobe"]),
        "drama_config": checkpoint is not None and (checkpoint / "config.json").is_file(),
        "drama_codec": checkpoint is not None and (checkpoint / "codec.pth").is_file(),
        "drama_index": checkpoint is not None and (checkpoint / "model.safetensors.index.json").is_file(),
        "whisper_config": (verifier / "config.json").is_file(),
        "whisper_weights": bool(list(verifier.glob("*.safetensors"))) or (verifier / "weights.npz").is_file(),
    }
    if faster_whisper:
        from .model_setup import faster_whisper_model_files
        checks.update({"whisper_" + name: present
                       for name, present in faster_whisper_model_files(verifier).items()})
    reference_model = config["verification"].get("reference_model")
    reference_verification = {"configured": reference_model is not None,
                              "model": reference_model, "available": None}
    if reference_model is not None:
        reference = Path(reference_model)
        checks["reference_asr_config"] = (reference / "config.json").is_file()
        checks["reference_asr_weights"] = any(
            candidate.is_file() for candidate in reference.glob("*.safetensors"))
        reference_verification["available"] = (
            checks["reference_asr_config"] and checks["reference_asr_weights"])
        if faster_whisper:
            reference_files = faster_whisper_model_files(reference)
            checks.update({"reference_asr_" + name: present for name, present in reference_files.items()})
            reference_verification["available"] = all(reference_files.values())
    if config.get("voice_design", {}).get("type") == "qwen3_mlx":
        designer = Path(config["voice_design"]["checkpoint"])
        checks["voice_design_weights"] = bool(list(designer.glob("*.safetensors")))
        checks["voice_design_tokenizer"] = (
            (designer / "speech_tokenizer" / "config.json").is_file()
            and bool(list((designer / "speech_tokenizer").glob("*.safetensors")))
        )
        try:
            checks["voice_design_config"] = json.loads(
                (designer / "config.json").read_text()).get("tts_model_type") == "voice_design"
        except (OSError, ValueError):
            checks["voice_design_config"] = False
    if checks["drama_index"]:
        try:
            index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
            checks["drama_shards"] = all((checkpoint / name).is_file() for name in set(index["weight_map"].values()))
        except (ValueError, KeyError, TypeError):
            checks["drama_shards"] = False
    else:
        checks["drama_shards"] = False
    model_problem = None
    if remote_only:
        checks = {key: value for key, value in checks.items() if not key.startswith("drama_")}
        dependencies.pop("drama3-mlx", None)
    elif config["engine"]["type"] != "drama3_mlx":
        from .engines import engine_entry_points

        checks = {key: value for key, value in checks.items() if not key.startswith("drama_")}
        dependencies.pop("drama3-mlx", None)
        matches = engine_entry_points(config["engine"]["type"])
        checks["tts_plugin_installed"] = len(matches) == 1
    else:
        selected = get_drama3_model(config["engine"].get("model", DEFAULT_DRAMA3_MODEL))
        if selected.runtime == "mlx_audio":
            from .model_setup import ModelSetupError, engine_checkpoint_missing

            dependencies.pop("drama3-mlx", None)
            checks = {key: value for key, value in checks.items() if not key.startswith("drama_")}
            checks["mlx_audio_drama3_version"] = supports_mlx_audio_drama3(dependencies["mlx-audio"])
            try:
                missing = engine_checkpoint_missing(config["engine"])
                checks["drama_checkpoint"] = not missing
                if missing:
                    model_problem = "Missing: " + ", ".join(missing)
            except (ModelSetupError, ValueError) as exc:
                checks["drama_checkpoint"] = False
                model_problem = str(exc)
        else:
            from .engines import DRAMA3_RUNTIME_VERSION

            checks["drama_runtime_version"] = dependencies.get("drama3-mlx") == DRAMA3_RUNTIME_VERSION
    if any(worker["type"] == "fish_http" for worker in workers):
        try:
            dependencies["msgpack"] = metadata.version("msgpack")
        except metadata.PackageNotFoundError:
            dependencies["msgpack"] = None
    return {"ready": all(checks.values()) and all(dependencies.values()),
            "checks": checks, "packages": dependencies,
            "model_problem": model_problem,
            "reference_verification": reference_verification,
            "note": ("Checks installed files; a real generation test is still needed to exercise the configured inference backends."
                     if faster_whisper else "Checks installed files; a real generation test is still needed to exercise Metal.")}


def main(argv=None):
    args = parser().parse_args(argv)
    engine = verifier = None
    try:
        config = (load_config(args.config, require_verifier=False) if args.command == "serve"
                  else load_config(args.config))
        if args.command == "serve":
            if config["engine"]["type"] != "drama3_mlx" or not config["engine"]["checkpoint"]:
                raise ValueError("serve requires engine.type = drama3_mlx and a local engine.checkpoint")
            from .model_setup import ModelSetup
            from .server import serve
            ModelSetup(config).ensure("engine")
            serve(config)
            return 0
        if args.command == "serve-chapters":
            from .chapter_server import serve
            serve(config)
            return 0
        if args.command == "doctor":
            report = doctor(config)
            print(json.dumps(report, indent=2))
            return 0 if report["ready"] else 2
        bundle = load_bundle(args.bundle)
        config = resolve_book_config(bundle, config, output=args.output,
                                     work_dir=args.work_dir, epub=args.epub)
        if args.command == "cleanup":
            from .workflow import cleanup_render_work
            result = cleanup_render_work(bundle, config, dry_run=args.dry_run,
                                         discard_render_cache=args.discard_render_cache)
            if args.dry_run:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            action = "Would remove" if args.dry_run else "Removed"
            print(f"{action} {result['files_removed']} file(s), "
                  f"{result['bytes_removed'] / (1024 * 1024):.1f} MiB.")
            print("Voice samples, casting candidates, and published audiobooks are preserved.")
            if args.discard_render_cache:
                print("Discarding the render cache removes speech resumability; future rendering will generate speech again.")
            else:
                print("Accepted speech and current chapter checkpoints are retained for resume.")
            return 0
        if getattr(args, "chapters", None) is not None:
            config["render"]["chapter_numbers"] = chapter_numbers(args.chapters, len(bundle["chapters"]))
        if getattr(args, "allow_review", None) is not None:
            config["render"]["allow_review"] = args.allow_review
        if getattr(args, "chapter_format", None) is not None:
            config["render"]["chapter_format"] = args.chapter_format
        if args.command == "voice-aliases":
            from .book_aliases import edit_book_aliases
            result = edit_book_aliases(bundle, config, assignments=args.alias_assignments,
                                       removals=args.alias_removals)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "cast":
            from .audition import casting_command
            config = select_voice_designer(config, args.voice_designer)
            return casting_command(args, bundle, config)
        plan = make_plan(bundle, config)
        if args.command == "plan":
            from .workflow import make_render_plan
            workers = config.get("render_pool", {}).get("workers", [])
            multi_speaker = (all(worker.get("multi_speaker", True) for worker in workers)
                             if workers else config["engine"]["type"] == "drama3_mlx")
            plan = make_render_plan(bundle, config, multi_speaker=multi_speaker)
            print(json.dumps({
                "book": plan["book"]["title"], "chapters": len(plan["chapters"]),
                "chunks": sum(len(c["chunks"]) for c in plan["chapters"]),
                "voices": len(plan["voice_ids"]), "partial_book": plan["partial_book"],
                "review_items": plan["review_items"], "errors": plan["errors"],
                "series_id": config["voices"]["series_id"],
                "voice_directory": config["voices"]["directory"],
                "voice_search_directories": [config["voices"]["directory"],
                                             *[str(Path(path) / "voices") for path in config["_voice_parents"]]],
                "work_directory": config["render"]["work_dir"],
                "output_directory": config["render"]["output_dir"],
                "renderers": [{key: worker[key] for key in ("name", "type", "concurrency")}
                              for worker in workers],
                "chapter_titles": [{"number": c["number"], "title": c["title"]} for c in plan["chapters"]],
            }, ensure_ascii=False, indent=2))
            return 2 if plan["errors"] else 0
        expected_series = bundle["manifest"].get("provenance", {}).get("series", {}).get("series_id")
        if expected_series and expected_series != config["voices"]["series_id"]:
            raise ValueError("Configured voices.series_id must match the analyzer's series ID")
        if args.command in {"render", "assemble"} and plan["errors"]:
            raise ValueError("\n".join(plan["errors"]))
        if args.command == "assemble":
            from .chapter_client import assemble_remote, has_remote_chapters
            if has_remote_chapters(config):
                assemble_remote(bundle, config)
            else:
                from .workflow import assemble
                assemble(bundle, config)
            print(f"Completed audiobook: {config['render']['output_dir']}")
            return 0
        if args.command == "render" and config.get("remote_chapters", {}).get("url"):
            from .chapter_client import render_remote
            result = render_remote(bundle, config, chapters_only=args.chapters_only)
            if args.chapters_only:
                for chapter in result.get("chapters", []):
                    print(f"Completed chapter {chapter['number']} / {len(bundle['chapters'])}: "
                          f"{chapter.get('audio_file') or 'empty chapter (no speech)'}")
            else:
                print(f"Completed audiobook: {config['render']['output_dir']}")
            return 0
        store = voice_store(bundle, config)
        if args.command == "voices":
            output = []
            for voice in bundle["cast"]["voices"]:
                saved = store.load(voice["id"])
                output.append({"voice_id": voice["id"], "description": voice["description"],
                               "saved": saved is not None,
                               "inherited": saved.get("inherited", False) if saved else False,
                               "voice_directory": saved.get("voice_directory") if saved else None,
                               "reference_audio": saved["reference_audio"] if saved else None,
                               "description_applied": saved.get("description_applied") if saved else None})
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0
        from .engines import create_engine
        from .verification import create_verifier
        from .workflow import ensure_voice, import_voice, render
        from .model_setup import ModelSetup
        setup = ModelSetup(config)
        workers = config.get("render_pool", {}).get("workers", []) if args.command == "render" else []
        remote_only = bool(workers) and all(worker["type"] != "local" for worker in workers)
        if not remote_only:
            setup.ensure("engine")
        setup.ensure("verification")
        if args.command in {"render", "create-voices"}:
            missing = [voice for voice in plan["voice_ids"] if store.load(voice) is None]
            if missing:
                if remote_only:
                    from .config import resolve_voice_design
                    if resolve_voice_design(config)["type"] == "engine":
                        raise ValueError("Missing voices need a voice designer: configure Gemini/ElevenLabs, "
                                         "add a local renderer, or prepare the voices with cast first")
                setup.ensure("voice_design")
                from epub_tts_narrator.audition_scripts import ensure_bundle_scripts
                bundle = ensure_bundle_scripts(bundle, args.config, canonical_id=store.canonical_id, voice_ids=missing)
        if workers:
            from .pool import create_render_engine
            engine = create_render_engine(config)
            print("Rendering with " + ", ".join(
                f"{worker['name']} ({worker['concurrency']} slot(s))" for worker in workers))
        else:
            engine = create_engine(config["engine"])
        verifier = create_verifier(config["verification"], config.get("jev"))
        if workers:
            verifier = engine.wrap_verifier(verifier)
        if args.command == "import-voice":
            result = import_voice(bundle, args.voice_id, args.audio,
                                  args.transcript.read_text(encoding="utf-8"), config, engine, verifier)
            print(f"Saved immutable voice {result['voice_id']}: {result['reference_audio']}")
        elif args.command == "create-voices":
            profiles = {voice["id"]: voice for voice in bundle["cast"]["voices"]}
            lock = store.lock()
            try:
                store.persist_aliases()
                for voice_id in plan["voice_ids"]:
                    profile = profiles.get(store.canonical_id(voice_id), profiles[voice_id])
                    ensure_voice(store, voice_id, profile, engine, verifier,
                                 config, config["render"]["work_dir"], bundle=bundle)
            finally:
                lock.rmdir()
            print("Persistent voices are saved. Audition their reference WAVs with the voices command.")
        else:
            result = render(bundle, config, engine, verifier, chapters_only=args.chapters_only)
            if args.chapters_only:
                print(f"Completed chapter audio: {Path(config['render']['work_dir']) / 'chapters'}")
                if isinstance(result, dict):
                    for chapter in result.get("chapters", []):
                        location = chapter.get("audio_file") or "empty chapter (no speech)"
                        print(f"  Chapter {chapter['number']} / {len(bundle['chapters'])}: {location}")
            else:
                print(f"Completed audiobook: {config['render']['output_dir']}")
        return 0
    except ProviderError as exc:
        print(f"Renderer stopped: {exc}", file=sys.stderr)
        print("Validated text-model responses remain cached. Rerun the same command to retry incomplete requests.",
              file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Renderer stopped: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; verified clips are retained for resume.", file=sys.stderr)
        return 130
    finally:
        for component in (verifier, engine):
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except (ValueError, RuntimeError, OSError) as exc:
                    print(f"Renderer cleanup warning: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
