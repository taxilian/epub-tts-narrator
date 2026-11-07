"""Check batched Drama 3 decoding against Fish's own single-request decoder.

Run inside an environment with torch and Fish's pinned ``fish_speech`` package
(the Runpod worker's Fish environment, or a scratch venv with Fish's source on
PYTHONPATH). It never loads book text and never writes model files.

``--tiny`` builds a small random-weight Drama 3 from Fish's own classes on the
CPU. ``--checkpoint DIR`` loads the real model as the worker does. Checks:

1. equivalence: Fish's ``generate_drama3`` one request at a time and the
   batched backend (through the real scheduler, fewer rows than requests so
   requests join mid-flight) receive identical forced sampler results; every
   logits tensor handed to the sampler and every frame must agree;
2. independence: with the real counter-hash sampler, each request's frames are
   identical whether it decodes alone or batched with others;
3. graphs (CUDA only): captured-graph decoding equals eager batched decoding;
4. speed (CUDA only): decode steps per second for 1 row and for --rows rows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tiny_model(dtype):
    import torch
    from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
    from fish_speech.models.text2semantic.fish_omni.config import (
        AudioDecoderConfig, FishOmniConfig, FishOmniTextConfig)
    from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder
    text = FishOmniTextConfig(
        vocab_size=160, hidden_size=64, num_hidden_layers=6,
        layer_types=["linear_attention", "linear_attention", "full_attention"] * 2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rope_theta=10000.0,
        partial_rotary_factor=0.25, rms_norm_eps=1e-6, max_position_embeddings=4096,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32)
    audio = AudioDecoderConfig(dim=64, intermediate_size=96, n_head=4, n_local_heads=2, head_dim=16,
                               norm_eps=1e-6, rope_base=10000.0, rotary_percent=0.5, vocab_size=32,
                               num_codebooks=4, shared_enrichment_layers=2, vq_backbone_codebooks=1)
    config = FishOmniConfig(text=text, audio_decoder=audio, semantic_start_token_id=100,
                            semantic_end_token_id=131, eos_token_id=7)
    model = FishOmniModel(config, FishOmniBackbone(text, dtype=dtype, device="cpu"),
                          FishOmniAudioDecoder(config, dtype=dtype, device="cpu"), tokenizer=None).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.normal_(0.0, 0.3 if "A_log" in name or "dt_bias" in name else 0.08)
    return model


def real_model(checkpoint, device):
    import torch
    from fish_speech.models.text2semantic.inference import init_model
    model, _ = init_model(str(checkpoint), device, torch.bfloat16)
    return model.eval()


def prompts(model, count, rng):
    """Synthetic prompt frames of different lengths; never book text."""
    import torch
    config = model.config
    width = 1 + config.audio_decoder.num_codebooks
    band = (config.semantic_start_token_id, config.semantic_end_token_id)
    out = []
    for rows in (37, 150, 5, 260, 18, 90)[:count]:
        frames = torch.zeros((rows, width), dtype=torch.long)
        for row in range(rows):
            if rng.random() < 0.5:
                frames[row, 0] = rng.randint(20, min(90, band[0] - 1))
            else:
                main = rng.randint(*band)
                frames[row, 0] = main
                frames[row, 1] = main - band[0]
                frames[row, 2:] = torch.randint(0, config.audio_decoder.vocab_size, (width - 2,))
        out.append(frames)
    return out


def scripts_for(model, prompt_count, rng):
    audio = model.config.audio_decoder
    lengths = (23, 9, 31, 17, 12, 27)[:prompt_count]
    result = []
    for steps in lengths:
        rows = []
        for step in range(steps):
            first = 0 if step == steps - 1 else rng.choice([rng.randint(1, audio.vocab_size), 5, 5])
            rows.append([first, rng.randint(1, audio.vocab_size)]
                        + [rng.randint(0, audio.vocab_size - 1) for _ in range(audio.num_codebooks - 1)])
        result.append(rows)
    return result


class ForcedHarness:
    """Wraps the backend: forced sampler results per request, logits recorded."""

    def __init__(self, inner, scripts):
        self.inner, self.rows, self.eos_id, self.scripts = inner, inner.rows, inner.eos_id, scripts
        self.owner, self.progress = {}, {}
        self.seen = {index: [] for index in range(len(scripts))}
        inner.sampler = self.sample

    def prefill(self, slot, request):
        self.owner[slot], self.progress[slot] = request.seed, 0
        self.inner.prefill(slot, request)

    def release(self, slot):
        self.owner.pop(slot, None)
        self.inner.release(slot)

    def sample(self, logits, temperature, top_k, top_p, draw):
        import torch
        values = []
        for slot in range(self.rows):
            if slot in self.owner:
                index = self.owner[slot]
                self.seen[index].append(logits[slot].detach().float().cpu())
                values.append(self.scripts[index][self.progress[slot]][draw])
            else:
                values.append(0)
        return torch.tensor(values, device=logits.device)

    def step(self, slots):
        result = self.inner.step(slots)
        for slot in slots:
            self.progress[slot] += 1
        return result


def run_batched(model, requests, rows, *, graphs, capacity, harness=None, seeds=None, max_new=60):
    import torch
    from epub_tts_narrator.renderer.drama3_batch import BatchedDrama3Backend
    from epub_tts_narrator.renderer.drama3_batch_scheduler import Drama3BatchScheduler
    with torch.inference_mode():
        backend = BatchedDrama3Backend(model, rows=rows, capacity=capacity, graphs=graphs)
    target = harness(backend) if harness else backend
    scheduler = Drama3BatchScheduler(target)
    seeds = seeds or list(range(len(requests)))
    futures = [scheduler.submit(frames, max_new, temperature=0.7, top_k=30, top_p=0.7, seed=seed)
               for frames, seed in zip(requests, seeds)]
    results = [future.result(timeout=1800) for future in futures]
    scheduler.close()
    return results, target, scheduler


def check_equivalence(model, requests, scripts, rows, capacity, tolerance):
    import torch
    from fish_speech.models.text2semantic.fish_omni import tower as tower_module
    from fish_speech.models.text2semantic.fish_omni.generate import generate_drama3
    reference = []
    real_sampler = tower_module.sample_top_k_top_p
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"  # a forced sampler cannot be replayed from a graph
    try:
        for frames, steps in zip(requests, scripts):
            calls, seen = [value for step in steps for value in step], []

            def forced(logits, temperature, top_k, top_p, *, check_finite=True):
                seen.append(logits.detach().float().cpu())
                return torch.tensor([calls[len(seen) - 1]], device=logits.device)
            tower_module.sample_top_k_top_p = forced
            with torch.inference_mode():
                out, finish = generate_drama3(model, frames, max_new_tokens=60,
                                              temperature=0.7, top_k=30, top_p=0.7)
            reference.append((out.tolist(), finish, seen))
    finally:
        tower_module.sample_top_k_top_p = real_sampler
        os.environ.pop("FISH_DRAMA3_CUDA_GRAPH", None)
    results, harness, scheduler = run_batched(
        model, requests, rows, graphs=False, capacity=capacity,
        harness=lambda backend: ForcedHarness(backend, scripts))
    worst, first = 0.0, None
    draws = 1 + model.config.audio_decoder.num_codebooks
    for index, ((ref_frames, ref_finish, ref_seen), (frames, finish)) in enumerate(zip(reference, results)):
        ours = harness.seen[index]
        for call, (a, b) in enumerate(zip(ref_seen, ours)):
            diff = (a.reshape(-1) - b.reshape(-1)).abs().max().item()
            relative = diff / max(a.abs().max().item(), 1e-6)
            worst = max(worst, relative)
            if relative > tolerance and first is None:
                first = {"request": index, "step": call // draws, "draw": call % draws, "relative": relative}
        if first is None and (frames != ref_frames or finish != ref_finish or len(ours) != len(ref_seen)):
            first = {"request": index, "frames": "differ"}
    if first is not None or worst > tolerance:
        raise AssertionError(f"batched decoding differs from Fish's decoder: first {first}, "
                             f"worst relative logits difference {worst:.2e} (limit {tolerance:.0e})")
    return {"requests": len(requests), "rows": rows, "steps": scheduler.steps,
            "row_steps": scheduler.row_steps, "worst_relative_logits_difference": worst}


def check_independence(model, requests, rows, capacity, graphs):
    seeds = [1729 + 977 * index for index in range(len(requests))]
    alone = [run_batched(model, [frames], 1, graphs=graphs, capacity=capacity, seeds=[seed])[0][0]
             for frames, seed in zip(requests, seeds)]
    together = run_batched(model, requests, rows, graphs=graphs, capacity=capacity, seeds=seeds)[0]
    identical = sum(a == b for a, b in zip(alone, together))
    return {"requests": len(requests), "identical_alone_vs_batched": identical}


def check_graphs(model, requests, rows, capacity):
    seeds = [4102 + 31 * index for index in range(len(requests))]
    eager = run_batched(model, requests, rows, graphs=False, capacity=capacity, seeds=seeds)[0]
    graphed = run_batched(model, requests, rows, graphs=True, capacity=capacity, seeds=seeds)[0]
    identical = sum(a == b for a, b in zip(eager, graphed))
    return {"requests": len(requests), "identical_graph_vs_eager": identical}


def measure_speed(model, rows, capacity, steps=200):
    import torch
    from epub_tts_narrator.renderer.drama3_batch import BatchedDrama3Backend
    from epub_tts_narrator.renderer.drama3_batch_scheduler import BatchRequest
    rng = random.Random(3)
    prompt = prompts(model, 1, rng)[0]
    result = {}
    for count in sorted({1, rows}):
        with torch.inference_mode():
            backend = BatchedDrama3Backend(model, rows=count, capacity=capacity, graphs=True)
        for slot in range(count):
            backend.prefill(slot, BatchRequest(prompt, steps + 1, 0.7, 30, 0.7, slot))
        slots = list(range(count))
        backend.step(slots)
        torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(steps):
            backend.step(slots)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        result[f"rows_{count}"] = {"steps_per_second": steps / elapsed,
                                   "row_frames_per_second": count * steps / elapsed}
        del backend
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", action="store_true")
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                        help="--tiny only; the real model always loads in bfloat16")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--requests", type=int, choices=range(2, 7), default=None,
                        help="Requests to decode (default: 3 with --tiny, 6 with --checkpoint)")
    args = parser.parse_args()
    import torch
    rng = random.Random(7)
    torch.manual_seed(7)
    if args.tiny:
        model, capacity = tiny_model(getattr(torch, args.dtype)), 512
        tolerance = 1e-4 if args.dtype == "float32" else 5e-2
    else:
        model, capacity = real_model(args.checkpoint, args.device), None
        tolerance = 5e-2
    count = args.requests or (3 if args.tiny else 6)
    requests = [frames.to(model.audio_decoder.device) for frames in prompts(model, count, rng)]
    scripts = scripts_for(model, count, rng)
    equivalence_rows = max(1, min(args.rows, count - 1))
    report = {"model": "tiny" if args.tiny else "checkpoint",
              "equivalence": check_equivalence(model, requests, scripts, equivalence_rows, capacity, tolerance)}
    print(json.dumps(report["equivalence"]), flush=True)
    cuda = model.audio_decoder.device.type == "cuda"
    report["independence"] = check_independence(model, requests, min(args.rows, count), capacity, graphs=cuda)
    print(json.dumps(report["independence"]), flush=True)
    if cuda:
        report["graphs"] = check_graphs(model, requests, min(args.rows, count), capacity)
        print(json.dumps(report["graphs"]), flush=True)
        report["speed"] = measure_speed(model, args.rows, capacity)
        print(json.dumps(report["speed"]), flush=True)
    print("BATCHING_REPORT=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
