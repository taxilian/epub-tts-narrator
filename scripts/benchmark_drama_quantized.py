"""Isolated voice-cloning comparison; does not change renderer config or voices.

Run each backend in its own process/environment, then verify in the renderer
environment. See scripts/QUANTIZED_TTS.md for setup and interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = [
    {"name": "narration", "turns": [
        {"speaker": 0, "text": "Lena and I watched the harbor wake. The lantern was still warm, and someone had left the door open."},
    ]},
    {"name": "dialogue", "turns": [
        {"speaker": 0, "text": "The lantern is still burning."},
        {"speaker": 1, "text": "Then we have time to find the path."},
    ]},
]


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packages():
    result = {}
    for name in ("mlx", "mlx-metal", "mlx-audio", "drama3-mlx", "numpy",
                 "soundfile", "mlx-whisper", "transformers"):
        try:
            dist = importlib.metadata.distribution(name)
            result[name] = {"version": dist.version}
            direct = dist.read_text("direct_url.json")
            if direct:
                result[name]["source"] = json.loads(direct)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def reference(directory):
    meta = json.loads((directory / "voice.json").read_text())
    wav = (directory / "reference.wav").resolve()
    digest = checksum(wav)
    if digest != meta["reference_sha256"]:
        raise ValueError(f"Reference checksum mismatch: {wav}")
    return {"kind": "reference", "reference_audio": str(wav),
            "reference_text": meta["reference_text"], "reference_sha256": digest}


def summarize(rows):
    """Include failed-attempt time; exclude warmups from steady-state totals."""
    measured = [row for row in rows if not row["warmup"]]
    elapsed = sum(row["elapsed_seconds"] for row in measured)
    audio = sum(row.get("duration_seconds", 0) for row in measured)
    accepted = [row for row in measured if row.get("sanity_passed") is True]
    accepted_audio = sum(row["duration_seconds"] for row in accepted)
    checked = all("sanity_passed" in row for row in measured)
    return {
        "measured_attempts": len(measured),
        "generation_errors": sum("error" in row for row in measured),
        "generation_seconds": elapsed,
        "audio_seconds": audio,
        "audio_seconds_per_wall_second": audio / elapsed if elapsed else None,
        "wall_seconds_per_audio_second": elapsed / audio if audio else None,
        "verification_complete": bool(measured) and checked,
        "sanity_passed": len(accepted) if checked else None,
        "accepted_audio_seconds_per_wall_second": accepted_audio / elapsed if checked and elapsed else None,
        "peak_generation_gb": max((row["peak_mlx_gb"] for row in rows), default=0),
    }


def generate(args):
    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    voices = [reference(directory) for directory in args.voice_dirs]
    # Refuse to mix runs or overwrite existing audio.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "manifest.json"
    report = {
        "backend": args.backend, "checkpoint": str(args.checkpoint.resolve()),
        "experts": args.experts if args.backend == "sogni" else "checkpoint-defined",
        "model_id": args.model_id, "model_revision": args.revision,
        "python": sys.version, "platform": platform.platform(), "packages": packages(),
        "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"], text=True).strip().splitlines(),
        "sampling": {"temperature": 1.0, "top_p": 0.9, "top_k": 30, "max_frames": args.max_frames},
        "seeds": args.seeds, "references": voices, "cases": CASES, "results": [],
        "timing": "Complete public synthesis call plus GPU synchronization; includes reference read/encoding/prefill on first call and WAV writing. One warmup per case excluded from steady-state totals. No ASR, download or model load in generation timing.",
        "limitations": "Different runtimes and quantization recipes; same seed does not imply identical samples. ASR cannot establish voice identity or acting quality. Peak MLX allocation is not whole-system RAM. No dollar-cost estimate.",
        "status": "loading",
    }
    save(manifest, report)
    mx.reset_peak_memory()
    started = time.perf_counter()
    if args.backend == "sogni":
        from epub_tts_narrator.renderer.engines import Drama3MLXEngine
        model = Drama3MLXEngine(args.checkpoint, experts=args.experts, max_new_tokens=args.max_frames)
        model.warmup()
    else:
        from mlx_audio.tts.utils import load_model
        model = load_model(str(args.checkpoint.resolve()))
    mx.synchronize()
    report.update(load_seconds=time.perf_counter() - started,
                  load_peak_mlx_gb=mx.get_peak_memory() / 1e9,
                  status="generating")
    save(manifest, report)
    print(f"Loaded {args.backend} in {report['load_seconds']:.2f}s", flush=True)

    for case in CASES:
        for take, seed in enumerate([0, *args.seeds]):
            name = f"{case['name']}-{'warmup' if take == 0 else f'seed-{seed}'}"
            path = args.output / f"{name}.wav"
            turns = [{"text": turn["text"], "voice": voices[turn["speaker"]]}
                     for turn in case["turns"]]
            expected = " ".join(turn["text"] for turn in turns)
            row = {"name": name, "case": case["name"], "seed": seed,
                   "warmup": take == 0, "expected_text": expected, "audio": path.name}
            mx.synchronize()
            mx.reset_peak_memory()
            started = time.perf_counter()
            try:
                if args.backend == "sogni":
                    if len(turns) == 1:
                        row.update(model.synthesize(expected, voices[0], path, seed=seed))
                    else:
                        row.update(model.synthesize_dialogue(turns, path, seed=seed))
                    row["hit_frame_cap"] = False  # adapter requires EOS to publish
                else:
                    mx.random.seed(seed)
                    if len(turns) == 1:
                        text = expected
                        ref_audio = voices[0]["reference_audio"]
                        ref_text = voices[0]["reference_text"]
                    else:
                        text = " ".join(f"<|speaker:{turn['speaker']}|> {turn['text']}" for turn in case["turns"])
                        ref_audio = [voice["reference_audio"] for voice in voices]
                        ref_text = [f"<|speaker:{i}|> {voice['reference_text']}" for i, voice in enumerate(voices)]
                    results = list(model.generate(text=text, ref_audio=ref_audio, ref_text=ref_text,
                                                  temperature=1.0, top_p=0.9, top_k=30,
                                                  max_tokens=args.max_frames, stream=False))
                    if not results:
                        raise ValueError("Model returned no audio")
                    samples = np.concatenate([np.asarray(result.audio, dtype=np.float32) for result in results])
                    if not len(samples) or not np.isfinite(samples).all():
                        raise ValueError("Model returned empty or nonfinite audio")
                    rate = model.sample_rate
                    sf.write(path, samples, rate, subtype="FLOAT")
                    capped = any(result.token_count >= args.max_frames for result in results)
                    row.update(duration_seconds=len(samples) / rate, sample_rate=rate,
                               sample_count=len(samples), token_counts=[result.token_count for result in results],
                               hit_frame_cap=capped,
                               finish_reason="frame_cap" if capped else "inferred_eos")
                    del results, samples
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}", sanity_passed=False)
            mx.synchronize()
            row.update(elapsed_seconds=time.perf_counter() - started,
                       peak_mlx_gb=mx.get_peak_memory() / 1e9)
            if path.exists():
                row["audio_sha256"] = checksum(path)
            report["results"].append(row)
            report["summary"] = summarize(report["results"])
            save(manifest, report)
            print(json.dumps({key: row[key] for key in ("name", "elapsed_seconds", "duration_seconds", "peak_mlx_gb", "error") if key in row}), flush=True)
    report["references_unchanged"] = all(checksum(Path(v["reference_audio"])) == v["reference_sha256"] for v in voices)
    report["status"] = "generated"
    save(manifest, report)


def verify(args):
    from epub_tts_narrator.renderer.assembly import normalize_clip
    from epub_tts_narrator.renderer.quality import MlxWhisperVerifier, validate_audio
    verifier = MlxWhisperVerifier(args.whisper, language="en")
    for manifest in args.manifests:
        report = json.loads(manifest.read_text())
        report["verification_model"] = verifier.identity
        for row in report["results"]:
            if "error" in row:
                row["sanity_passed"] = False
                continue
            path = manifest.parent / row["audio"]
            if checksum(path) != row["audio_sha256"]:
                raise ValueError(f"Generated audio changed: {path}")
            normalized = manifest.parent / "checked" / row["audio"]
            if normalized.exists():
                normalized.unlink()
            normalize_clip(path, normalized)
            row["verification_audio"] = str(normalized.relative_to(manifest.parent))
            row["verification_audio_sha256"] = checksum(normalized)
            result = validate_audio(normalized, row["expected_text"], verifier, {"max_wer": 0.0})
            row["verification"] = result
            row["sanity_passed"] = result["passed"] and not row["hit_frame_cap"]
            save(manifest, report)
            print(row["name"], row["sanity_passed"], result.get("metrics", {}).get("wer"), flush=True)
        report["summary"] = summarize(report["results"])
        report["status"] = "verified"
        save(manifest, report)
        print(json.dumps(report["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("generate")
    run.add_argument("--backend", choices=["sogni", "mlx-audio"], required=True)
    run.add_argument("--experts", choices=["q8", "q4"], default="q8",
                     help="Sogni routed-expert precision; MLX-Audio uses its checkpoint recipe")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--voice-dirs", type=Path, nargs=2, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seeds", type=int, nargs="+", default=[1729, 1730, 1731])
    run.add_argument("--max-frames", type=int, default=600)
    check = commands.add_parser("verify")
    check.add_argument("manifests", type=Path, nargs="+")
    check.add_argument("--whisper", required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if not 1 <= args.max_frames <= 2048:
            parser.error("max-frames must be from 1 through 2048")
        if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s <= 0xFFFFFFFF for s in args.seeds):
            parser.error("seeds must be distinct unsigned 32-bit integers")
        generate(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
