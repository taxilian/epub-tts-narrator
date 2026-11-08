"""Measure batched MLX Drama 3 throughput at several row counts on this Mac.

``.venv-renderer/bin/python scripts/benchmark_mlx_batch_rows.py --checkpoint models/drama-3 --rows 1,2,4,8``
loads the model once and, for each row count, decodes ``2 x rows`` (at least
8) requests through the real batch scheduler. Prompts have realistic shapes
(a reference plus two history turns, about 730 prompt rows, like a chapter
take) built from synthetic text and random codes, so no book text or voice is
read and the prompt cache stays cold, as for a voice's first takes. Each row
count prints generated frames per second, frames per second per row, mean
rows per step and the share of decode-thread time spent prefilling. Pick the
``server.batch_rows`` (or local worker ``concurrency``) past which frames per
second stops rising. It uses the GPU; nothing is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prompts(model, count, seed=7):
    import numpy as np
    from drama3_mlx.prompt import build_prompt
    rng = np.random.default_rng(seed)
    config = model.model.config
    codebooks, vocab = config.num_codebooks, config.audio_decoder.vocab_size

    def codes(frames):
        return rng.integers(0, vocab, size=(codebooks, frames)).astype(np.int64)

    sentence = "<|speaker:0|> The lantern swung once and went still, and nobody in the room said a word for a while."
    return [build_prompt(sentence, model.tokenizer, config, [sentence], [codes(320)],
                         [(sentence, codes(150)), (sentence, codes(140))]) for _ in range(count)]


def measure(model, rows, requests, max_new):
    from epub_tts_narrator.renderer.drama3_batch_scheduler import Drama3BatchScheduler
    from epub_tts_narrator.renderer.mlx_drama3_batch import BatchedMLXBackend
    scheduler = Drama3BatchScheduler(BatchedMLXBackend(model.model, rows=rows))
    try:
        scheduler.submit(requests[0], 4, temperature=0.7, top_k=30, top_p=0.7, seed=999).result()  # warm up
        before = scheduler.stats()
        began = time.perf_counter()
        futures = [scheduler.submit(prompt, max_new, temperature=0.7, top_k=30, top_p=0.7, seed=index)
                   for index, prompt in enumerate(requests)]
        frames = sum(len(future.result()[0]) for future in futures)
        wall = time.perf_counter() - began
        after = scheduler.stats()
    finally:
        scheduler.close()
    steps = after["steps"] - before["steps"]
    return {"rows": rows, "frames_per_second": round(frames / wall, 1),
            "frames_per_second_per_row": round(frames / wall / rows, 1),
            "mean_rows": round((after["row_steps"] - before["row_steps"]) / steps, 2) if steps else 0.0,
            "prefill_share": round((after["prefill_seconds"] - before["prefill_seconds"]) / wall, 3),
            "wall_seconds": round(wall, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experts", default="q8")
    parser.add_argument("--rows", default="1,2,4,8", help="comma-separated row counts")
    parser.add_argument("--max-new", type=int, default=200, help="frames per request at most")
    args = parser.parse_args()
    from epub_tts_narrator.renderer.engines import Drama3MLXEngine
    model = Drama3MLXEngine(args.checkpoint, experts=args.experts)._load()
    counts = [int(value) for value in args.rows.split(",")]
    report = []
    for rows in counts:
        result = measure(model, rows, prompts(model, max(8, 2 * rows)), args.max_new)
        report.append(result)
        print(f"{rows:>3} rows: {result['frames_per_second']:7.1f} frames/s "
              f"({result['frames_per_second_per_row']:5.1f} per row), {result['mean_rows']:5.2f} rows per step, "
              f"prefill {100 * result['prefill_share']:4.1f}% of time", flush=True)
    print("MLX_BATCH_ROWS_REPORT=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
