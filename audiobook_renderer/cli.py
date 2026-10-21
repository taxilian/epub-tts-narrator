"""A separate CLI for verified local audiobook rendering."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

from .bundle import load_bundle, make_plan
from .config import load_config
from .book_context import resolve_book_config, voice_store


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("plan", "render", "voices", "create-voices", "import-voice", "cast", "voice-aliases"):
        command = commands.add_parser(name)
        command.add_argument("bundle", type=Path)
        command.add_argument("--config", type=Path, help="Global TOML file (default: ./epub-audiobook.toml)")
        command.add_argument("--work-dir", type=Path, help="Book-specific cache directory (default: BOOK/render-work)")
        command.add_argument("-o", "--output", type=Path, help="Book-specific audio output (default: BOOK/audiobook)")
        command.add_argument("--epub", type=Path, help="Original EPUB for audiobook cover extraction")
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
            command.add_argument("--action", default="interactive",
                                 choices=("interactive", "list", "prepare-samples", "normalize-samples", "generate", "generate-missing", "regenerate-all", "edit", "play", "accept", "recover"),
                                 help="Default: interactive voice casting; other actions support scripting")
            command.add_argument("--voice-id")
            command.add_argument("--candidate-id")
            command.add_argument("--prompt-file", type=Path, help="UTF-8 voice description for the next generation")
            command.add_argument("--text-file", type=Path, help="UTF-8 spoken sample text for the next generation")
            command.add_argument("--attempt-file", type=Path,
                                 help="Saved attempt JSON to recheck without resynthesis (--action recover)")
            command.add_argument("--keep-unselected", action="store_true",
                                 help="Save a recovered take without changing the active voice (--action recover)")
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--config", type=Path, help="Global TOML file (default: ./epub-audiobook.toml)")
    return root


def doctor(config):
    dependencies = {}
    for name in ("drama3-mlx", "mlx", "mlx-whisper", "mlx-audio"):
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
    if config["engine"]["type"] != "drama3_mlx":
        checks = {key: value for key, value in checks.items() if not key.startswith("drama_")}
        dependencies.pop("drama3-mlx")
        matches = metadata.entry_points(group="audiobook_renderer.tts").select(name=config["engine"]["type"])
        checks["tts_plugin_installed"] = len(matches) == 1
    return {"ready": all(checks.values()) and all(dependencies.values()),
            "checks": checks, "packages": dependencies,
            "reference_verification": reference_verification,
            "note": "Checks installed files; a real generation test is still needed to exercise Metal."}


def main(argv=None):
    args = parser().parse_args(argv)
    engine = None
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            report = doctor(config)
            print(json.dumps(report, indent=2))
            return 0 if report["ready"] else 2
        bundle = load_bundle(args.bundle)
        config = resolve_book_config(bundle, config, output=args.output,
                                     work_dir=args.work_dir, epub=args.epub)
        if args.command == "voice-aliases":
            from .book_aliases import edit_book_aliases
            result = edit_book_aliases(bundle, config, assignments=args.alias_assignments,
                                       removals=args.alias_removals)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "cast":
            from .audition import casting_command
            return casting_command(args, bundle, config)
        plan = make_plan(bundle, config)
        if args.command == "plan":
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
                "chapter_titles": [{"number": c["number"], "title": c["title"]} for c in plan["chapters"]],
            }, ensure_ascii=False, indent=2))
            return 2 if plan["errors"] else 0
        expected_series = bundle["manifest"].get("provenance", {}).get("series", {}).get("series_id")
        if expected_series and expected_series != config["voices"]["series_id"]:
            raise ValueError("Configured voices.series_id must match the analyzer's series ID")
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
        if args.command == "render" and plan["errors"]:
            raise ValueError("\n".join(plan["errors"]))
        from .engines import create_engine
        from .verification import create_verifier
        from .workflow import ensure_voice, import_voice, render
        engine = create_engine(config["engine"])
        verifier = create_verifier(config["verification"])
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
            render(bundle, config, engine, verifier)
            print(f"Completed audiobook: {config['render']['output_dir']}")
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Renderer stopped: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; verified clips are retained for resume.", file=sys.stderr)
        return 130
    finally:
        if engine is not None and hasattr(engine, "close"):
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
