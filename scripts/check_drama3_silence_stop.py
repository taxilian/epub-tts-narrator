"""Check the watched Drama 3 decode loop against Fish's own ``generate_drama3``.

Run on the Runpod worker inside Fish's environment (torch + the pinned
``fish_speech``): ``python scripts/check_drama3_silence_stop.py --checkpoint
/workspace/models/drama-3``. It never loads book text and never writes model
files. Synthetic prompts (from ``check_drama3_batching``) are decoded with the
same seed three ways:

1. Fish's ``generate_drama3``;
2. ``fish_drama3_decode.generate_drama3_watched`` without a watch;
3. the same with the engine's real ``SilenceWatch`` (real codec decodes).

Frames and finish reasons must be identical, except that a take the watch
stopped must be an exact prefix of Fish's take. It also reports the time the
watch spent decoding as a share of decoding time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class TimedWatch:
    """Wraps a watch and times its decodes."""

    def __init__(self, watch):
        self.watch, self.seconds = watch, 0.0

    def observe(self, codes):
        import torch
        torch.cuda.synchronize()
        began = time.perf_counter()
        stop = self.watch.observe(codes)
        torch.cuda.synchronize()
        self.seconds += time.perf_counter() - began
        return stop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=400)
    args = parser.parse_args()
    import torch
    from fish_speech.models.text2semantic.fish_omni.generate import generate_drama3
    from fish_speech.utils import set_seed
    import check_drama3_batching as batching
    from epub_tts_narrator.renderer.fish_drama3_decode import generate_drama3_watched
    from epub_tts_narrator.renderer.fish_speech_engine import FishSpeechCudaEngine

    engine = FishSpeechCudaEngine(args.checkpoint, device=args.device)
    model, codec = engine._load()
    requests = batching.prompts(model, 6, random.Random(7))
    options = dict(max_new_tokens=args.max_new, temperature=0.7, top_k=30, top_p=0.7)
    report = {"cases": [], "identical": True}
    # Fish captures its CUDA graphs during the first decode of a process; warm
    # them up first so every compared decode starts from the same state.
    set_seed(12345)
    with torch.inference_mode():
        generate_drama3(model, requests[0].to(model.audio_decoder.device), **options)
    decode_seconds = generate_seconds = 0.0
    for index, prompt in enumerate(requests):
        prompt = prompt.to(model.audio_decoder.device)
        for seed in range(args.seeds):
            runs = {}
            for name in ("fish", "ours", "watched"):
                watch = TimedWatch(engine._silence_watch("Synthetic check text.", codec)) if name == "watched" else None
                set_seed(seed)
                torch.cuda.synchronize()
                began = time.perf_counter()
                with torch.inference_mode():
                    if name == "fish":
                        frames, finish = generate_drama3(model, prompt, **options)
                    else:
                        frames, finish = generate_drama3_watched(model, prompt, watch=watch, **options)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                runs[name] = (frames, finish)
                if watch is not None:
                    decode_seconds += watch.seconds
                    generate_seconds += elapsed
                    runs["decodes"] = watch.watch.decodes
            fish, ours, watched = runs["fish"], runs["ours"], runs["watched"]
            same = torch.equal(fish[0], ours[0]) and fish[1] == ours[1]
            if watched[1] == "silence":
                same_watched = torch.equal(watched[0], fish[0][: len(watched[0])])
            else:
                same_watched = torch.equal(watched[0], fish[0]) and watched[1] == fish[1]
            case = {"prompt": index, "seed": seed, "frames": len(fish[0]), "finish": fish[1],
                    "watched_frames": len(watched[0]), "watched_finish": watched[1],
                    "decodes": runs["decodes"], "identical": same, "watched_consistent": same_watched}
            report["identical"] &= same and same_watched
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
    report["watch_decode_share"] = round(decode_seconds / generate_seconds, 4) if generate_seconds else None
    print("SILENCE_STOP_REPORT=" + json.dumps(report), flush=True)
    sys.exit(0 if report["identical"] else 1)


if __name__ == "__main__":
    main()
