"""Reproducible, isolated Drama 3 prompt experiments; never changes a voice store."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON experiment configuration")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "config": config, "results": [],
        "note": "Research recordings only. ASR checks words; pitch does not verify perceived gender or identity.",
    }
    if manifest["config"] != config:
        raise ValueError("Use a fresh output directory when changing experiment settings")

    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from drama3_mlx import Drama3
    from drama3_mlx.generate import generate_frames
    from drama3_mlx.prompt import build_prompt

    print("Loading local Drama 3", flush=True)
    model = Drama3.load(config["checkpoint"], experts=config.get("experts", "q8"))
    decode_model = model.model
    if "main_top_k" in config:
        from audiobook_renderer.drama3_sampling import wrap_model
        decode_model = wrap_model(model.model, main_top_k=config["main_top_k"],
                                  codebook_top_k=config.get("top_k", 30))
    print("Model loaded", flush=True)
    for case in config["cases"]:
        for seed in config["seeds"]:
            case_id = f"{case['id']}-{seed}"
            if any(row["id"] == case_id for row in manifest["results"]):
                continue
            if "system" in case:
                literals = ["<|im_start|>system\n", case["system"], "<|im_end|>\n",
                            "<|im_start|>user\n", case["input"], "<|im_end|>\n",
                            "<|im_start|>assistant\n<|voice|>"]
                parts = []
                for literal in literals:
                    ids = model.tokenizer.encode(literal).ids
                    part = np.zeros((len(ids), 1 + model.model.config.num_codebooks), dtype=np.int64)
                    part[:, 0] = ids
                    parts.append(part)
                prompt = np.concatenate(parts)
            else:
                prompt = build_prompt(case["input"], model.tokenizer, model.model.config)
                literals = None
            print(f"Generating {case_id} ({len(prompt)} prompt tokens)", flush=True)
            started = time.perf_counter()
            mx.random.seed(seed)
            frames, finish = generate_frames(
                decode_model, prompt, max_new_tokens=config.get("max_new_tokens", 800),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 0.7), top_k=config.get("top_k", 30))
            if not len(frames):
                raise RuntimeError(f"{case_id} generated no frames ({finish})")
            codes = frames[:, 1:].T.copy()
            audio = model.codec.decode(codes)
            path = output / f"{case_id}.wav"
            if path.exists():
                raise FileExistsError(path)
            sf.write(path, audio, model.sample_rate, subtype="FLOAT")
            np.save(output / f"{case_id}.codes.npy", codes)
            row = {"id": case_id, "case": case, "seed": seed, "audio_path": str(path),
                   "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "prompt_literals": literals, "prompt_token_ids": prompt[:, 0].tolist(),
                   "sample_rate": model.sample_rate, "duration_seconds": len(audio)/model.sample_rate,
                   "finish_reason": finish, "frames": len(frames),
                   "generation_seconds": time.perf_counter()-started}
            manifest["results"].append(row)
            manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
            print(f"Saved {case_id}: {row['duration_seconds']:.2f}s, {finish}, {row['generation_seconds']:.1f}s elapsed", flush=True)
            mx.clear_cache()
    del decode_model, model
    gc.collect()
    mx.clear_cache()
    if config.get("whisper_model"):
        from audiobook_renderer.quality import MlxWhisperVerifier, compare_words
        verifier = MlxWhisperVerifier(config["whisper_model"], language="en")
        for row in manifest["results"]:
            if "transcription" in row:
                continue
            print(f"Transcribing {row['id']}", flush=True)
            transcript = verifier.transcribe_reference(Path(row["audio_path"]))
            row["transcription"] = transcript
            row["comparison"] = compare_words(row["case"]["expected_text"], transcript["text"])
            manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
            print(f"  WER {row['comparison'].get('wer')}: {transcript['text']}", flush=True)
        close = getattr(verifier, "close", None)
        if callable(close):
            close()
    print(f"Results: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
