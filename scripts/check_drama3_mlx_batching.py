"""Check the batched MLX Drama 3 decoder (``mlx_drama3_batch``) against the vendored loop.

``--tiny`` (no GPU, no model files): a small Drama 3 built from the vendored
classes with random weights, on MLX's CPU device. ``--checkpoint DIR`` uses a
real checkpoint on the default device (the GPU); ask before running it while
a render is using the machine. Three checks:

1. Teacher-forced logits: every row of the batched decoder, fed the frames
   the vendored ``generate_frames`` sampled for it, gives the same backbone
   and codebook-head logits per step as the vendored modules decoding that
   request alone, while rows of different lengths join and leave the batch.
   On the CPU (``--tiny``) multi-row kernels are exact, so this is exact.
   On the GPU bfloat16 multi-row kernels differ from one-row ones, so a
   ``--checkpoint`` run requires one row to be exact and staggered rows to
   drift no more than rows of one repeated request (measured 2026-11-07 on
   drama-3 q8: worst KL 0.14 at temperature 0.7 either way).
2. Row independence (``--tiny``): a request sampled inside a batch gives
   exactly the frames it gives alone.
3. The sampling noise equals the CUDA backend's counter-hash stream
   (a pure-Python splitmix64 reference), and the sampler's distribution
   matches the vendored sampler's.

Prints ``MLX_BATCH_REPORT=<json>`` and exits 1 on any failure. It never
loads book text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TINY_CONFIG = {'audio_decoder_config': {'attention_o_bias': False,
                              'attention_output_gate': True,
                              'attention_qk_norm': False,
                              'attention_qkv_bias': False,
                              'audio_embed_dim': None,
                              'audio_hidden_dim': 64,
                              'chunk_size_feed_forward': 0,
                              'codebook_loss_weight_per_layer': None,
                              'decoder_type': 'codebook_mtp_shared',
                              'dim': 64,
                              'dropout': 0.0,
                              'dtype': None,
                              'head_dim': 16,
                              'initializer_range': 0.02209708691207961,
                              'intermediate_size': 96,
                              'is_encoder_decoder': False,
                              'layernorm_zero_centered_gamma': True,
                              'max_seq_len': 11,
                              'model_type': 'fish_qwen3_audio_decoder',
                              'moe_intermediate_size': 32,
                              'mtp_spec': {'attention_variant': 'full_attention'},
                              'mtp_use_swa': False,
                              'n_head': 4,
                              'n_layer': 2,
                              'n_local_heads': 2,
                              'norm_eps': 1e-06,
                              'norm_topk_prob': True,
                              'num_codebooks': 4,
                              'num_experts': 1,
                              'num_experts_per_tok': 1,
                              'output_attentions': False,
                              'output_hidden_states': False,
                              'problem_type': None,
                              'return_dict': True,
                              'rope_base': 10000000.0,
                              'rotary_interleaved': True,
                              'rotary_percent': 0.25,
                              'router_gamma': 0.001,
                              'shared_enrichment_layers': 2,
                              'text_dim': 64,
                              'tie_word_embeddings': True,
                              'use_aux_loss_free': False,
                              'use_gradient_checkpointing': True,
                              'use_moe': False,
                              'vocab_size': 64,
                              'vq_backbone_codebooks': 1,
                              'vq_feature_placement': 'backbone_partial',
                              'vq_feature_scope': 'all_vq',
                              'window_attn_skip_freq': None,
                              'window_rotary_base': None,
                              'window_size': None},
     'audio_encoder_config': None,
     'audio_pad_token_id': 98,
     'dtype': 'bfloat16',
     'eos_token_id': 100,
     'fish_schema_version': 1,
     'model_type': 'fish_omni',
     'pad_token_id': 99,
     'quantization_config': {'activation_scheme': 'dynamic',
                             'quant_method': 'bsgemm-moe',
                             'weight_block_size': [128, 128]},
     'semantic_end_token_id': 263,
     'semantic_start_token_id': 200,
     'speech_template_version': 'reference-bracket-v1',
     'text_config': {'attention_bias': False,
                     'attention_dropout': 0.0,
                     'attn_output_gate': True,
                     'bos_token_id': None,
                     'chunk_size_feed_forward': 0,
                     'dtype': None,
                     'eos_token_id': 100,
                     'head_dim': 16,
                     'hidden_act': 'silu',
                     'hidden_size': 64,
                     'initializer_range': 0.02,
                     'intermediate_size': 32,
                     'is_encoder_decoder': False,
                     'layer_types': ['linear_attention',
                                     'linear_attention',
                                     'linear_attention',
                                     'full_attention'],
                     'layernorm_zero_centered_gamma': True,
                     'linear_conv_kernel_dim': 4,
                     'linear_key_head_dim': 16,
                     'linear_num_key_heads': 2,
                     'linear_num_value_heads': 4,
                     'linear_value_head_dim': 16,
                     'mamba_ssm_dtype': 'float32',
                     'max_position_embeddings': 4096,
                     'mlp_layer_types': ['moe', 'moe', 'moe', 'moe'],
                     'model_type': 'fish_omni_text',
                     'moe_intermediate_size': 32,
                     'num_attention_heads': 4,
                     'num_experts': 4,
                     'num_experts_per_tok': 2,
                     'num_hidden_layers': 4,
                     'num_key_value_heads': 2,
                     'output_attentions': False,
                     'output_hidden_states': False,
                     'pad_token_id': 99,
                     'problem_type': None,
                     'return_dict': True,
                     'rms_norm_eps': 1e-06,
                     'rope_parameters': {'partial_rotary_factor': 0.25,
                                         'rope_theta': 10000000.0,
                                         'rope_type': 'default'},
                     'rotary_interleaved': False,
                     'shared_expert_gate': True,
                     'shared_expert_intermediate_size': 32,
                     'tie_word_embeddings': False,
                     'use_cache': True,
                     'use_qk_norm': True,
                     'vocab_size': 600}}
_MASK = (1 << 64) - 1


def python_noise(seed, step, draw, width):
    """Pure-Python splitmix64 reference for ``sampling_noise``."""
    import math

    def mix(x):
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _MASK
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _MASK
        return x ^ (x >> 31)

    golden = 0x9E3779B97F4A7C15
    key = mix((seed * golden + mix((step * 64 + draw + 1) & _MASK)) & _MASK)
    out = []
    for column in range(width):
        bits = mix((key + column * golden) & _MASK) >> 41
        out.append(-math.log((bits + 0.5) / (1 << 23)))
    return out


def tiny_model(seed=0):
    import mlx.core as mx
    from drama3_mlx.config import Drama3Config
    from drama3_mlx.loading import Drama3Model
    mx.random.seed(seed)
    model = Drama3Model(Drama3Config.from_dict(TINY_CONFIG), fp8=False)
    mx.eval(model.parameters())
    return model.eval()


def random_prompt(model, rows, rng):
    """Prompt rows [rows, 1 + codebooks]: text tokens then semantic frames with codes."""
    import numpy as np
    config = model.config
    prompt = np.zeros((rows, 1 + config.num_codebooks), dtype=np.int64)
    text = rows // 3
    prompt[:text, 0] = rng.integers(0, config.eos_token_id - 2, size=text)
    prompt[text:, 0] = rng.integers(config.semantic_start_token_id, config.semantic_end_token_id + 1, size=rows - text)
    prompt[text:, 1:] = rng.integers(0, config.audio_decoder.vocab_size, size=(rows - text, config.num_codebooks))
    prompt[text:, 1] = prompt[text:, 0] - config.semantic_start_token_id
    return prompt


def reference_logits(model, prompt, frames):
    """The vendored modules decoding one request alone, teacher-forced with ``frames``."""
    import numpy as np
    import mlx.core as mx
    from drama3_mlx.generate import PREFILL_CHUNK_SIZE, run_backbone
    backbone, tower = model.backbone, model.tower
    caches, tower_caches = backbone.make_cache(), tower.make_caches()
    rows = mx.array(prompt.astype(np.int32))
    for start in range(0, len(prompt), PREFILL_CHUNK_SIZE):
        end = min(start + PREFILL_CHUNK_SIZE, len(prompt))
        hidden = run_backbone(model, rows[start:end], caches)
        paired_end = min(end, len(prompt) - 1)
        if paired_end > start:
            tower.forced(hidden[: paired_end - start], rows[start:paired_end], rows[start + 1: paired_end + 1], tower_caches)
    hidden, frame, out = hidden[-1:], rows[-1:], []
    shared = len(tower.shared_layers)
    for t in range(len(frames)):
        logits = backbone.lm_head(hidden)
        given = mx.array(frames[t:t + 1].astype(np.int32))
        main = given[:, 0]
        semantic = tower.is_semantic(main)
        c0 = mx.clip(main - tower.semantic_begin, 0, tower.cfg.vocab_size - 1)
        h = tower.run_shared(hidden, c0, semantic, frame, tower_caches)
        heads = []
        for index, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
            h = layer(h, given[:, 1 + index], semantic, tower_caches[shared + index])
            heads.append(head(h))
        out.append((logits[0], mx.stack(heads, axis=1)[0]))
        mx.eval(out[-1])
        hidden, frame = run_backbone(model, given, caches), given
    return out


class Request:
    def __init__(self, prompt, max_new_tokens, seed, temperature=0.7, top_k=30, top_p=0.7):
        self.prompt, self.max_new_tokens, self.seed = prompt, max_new_tokens, seed
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p


def _kl(want, got, temperature=0.7):
    """KL(want || got) between the two rows' softmax distributions at the sampling temperature."""
    import numpy as np
    want, got = want / temperature, got / temperature
    p = np.exp(want - want.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    q = np.exp(got - got.max(-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
    return float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum(-1).max())


def check_forced(model, prompts, frames, tolerance=None, *, together=False):
    """Teacher-forced batched logits against the vendored modules decoding each request alone.

    Rows join at different times (or all at once with ``together``) and leave
    when their frames run out. Returns the worst relative logit error and the
    worst per-row KL divergence of the sampling distributions.
    """
    import numpy as np
    import mlx.core as mx
    from epub_tts_narrator.renderer.mlx_drama3_batch import BatchedMLXBackend
    reference = [reference_logits(model, p, f) for p, f in zip(prompts, frames)]
    backend = BatchedMLXBackend(model, rows=len(prompts))
    done = {slot: 0 for slot in range(len(prompts))}
    worst = worst_kl = 0.0
    pending = list(range(len(prompts)))
    while pending and (together or not backend.order):
        slot = pending.pop(0)
        backend.prefill(slot, Request(prompts[slot], len(frames[slot]), slot))
    while backend.order:
        if pending and done[backend.order[0]] in (2, 5):
            slot = pending.pop(0)
            backend.prefill(slot, Request(prompts[slot], len(frames[slot]), slot))
        slots = list(backend.order)
        result = backend.forced_step(slots, {slot: frames[slot][done[slot]].tolist() for slot in slots})
        for slot in slots:
            want_main, want_heads = reference[slot][done[slot]]
            got_main, got_heads = result[slot]
            for want, got in ((want_main, got_main), (want_heads, got_heads)):
                # NumPy cannot read bfloat16 (the real model's dtype); widen in MLX first.
                want = np.array(want.astype(mx.float32), dtype=np.float64)
                got = np.array(got.astype(mx.float32), dtype=np.float64)
                worst = max(worst, float(np.abs(want - got).max() / max(1e-6, np.abs(want).max())))
                worst_kl = max(worst_kl, _kl(want, got))
            done[slot] += 1
            if done[slot] == len(frames[slot]):
                backend.release(slot)
    result = {"max_relative_logit_error": worst, "max_kl": worst_kl}
    if tolerance is not None:
        result["passed"] = worst <= tolerance
    return result


def check_gpu_drift(model, prompts, frames):
    """Real model on the GPU: one row is exact; several rows drift only as much as identical rows do.

    bfloat16 kernels compute a multi-row step differently from a one-row
    step, so batched rows cannot match the one-request decode exactly. Rows of
    one repeated request joined together have no padding and identical
    positions, so their drift is that kernel effect alone; staggered rows of
    different lengths must not drift measurably more, or padding/positions
    are wrong.
    """
    single = check_forced(model, prompts[:1], frames[:1], 1e-6)
    identical = check_forced(model, [prompts[0]] * len(prompts), [frames[0]] * len(prompts), together=True)
    staggered = check_forced(model, prompts, frames)
    passed = single["passed"] and staggered["max_kl"] <= 2 * identical["max_kl"] + 0.02
    return {"single_row": single, "identical_rows": identical, "staggered_rows": staggered, "passed": passed}


def check_independence(model, prompts, max_new):
    from epub_tts_narrator.renderer.drama3_batch_scheduler import Drama3BatchScheduler
    from epub_tts_narrator.renderer.mlx_drama3_batch import BatchedMLXBackend

    def run(indices, rows):
        scheduler = Drama3BatchScheduler(BatchedMLXBackend(model, rows=rows, run=runner))
        try:
            futures = {i: scheduler.submit(prompts[i], max_new, temperature=0.7, top_k=30, top_p=0.7, seed=100 + i)
                       for i in indices}
            return {i: future.result(timeout=600) for i, future in futures.items()}
        finally:
            scheduler.close()

    alone = {}
    for i in range(len(prompts)):
        alone.update(run([i], 1))
    together = run(list(range(len(prompts))), len(prompts))
    same = all(alone[i] == together[i] for i in alone)
    return {"passed": same, "frames": {i: len(alone[i][0]) for i in alone},
            "finish": {i: alone[i][1] for i in alone}}


def check_sampler(trials):
    import numpy as np
    import mlx.core as mx
    from drama3_mlx.sampling import sample_top_k_top_p
    from epub_tts_narrator.renderer.mlx_drama3_batch import sample_rows, sampling_noise
    seeds = mx.array(np.array([0, 1, 2**63 + 5, 2**64 - 1], dtype=np.uint64))
    steps = mx.array([0, 7, 123456, 3], dtype=mx.uint64)
    got = np.array(sampling_noise(seeds, steps, 3, 9))
    want = np.array([python_noise(s, t, 3, 9) for s, t in ((0, 0), (1, 7), (2**63 + 5, 123456), (2**64 - 1, 3))])
    noise_ok = bool(np.allclose(got, want, rtol=1e-6, atol=0) and (got > 0).all())
    # Flat enough that top-k 30 / top-p 0.9 keep many tokens, so the comparison means something.
    logits = mx.array(np.random.default_rng(1).normal(size=(1, 50)).astype(np.float32) * 0.6)
    ours = np.zeros(50)
    vendored = np.zeros(50)
    rows = mx.broadcast_to(logits, (trials, 50))
    picks = sample_rows(rows, mx.full((trials,), 0.7), mx.full((trials,), 30, dtype=mx.int32), mx.full((trials,), 0.9),
                        sampling_noise(mx.arange(trials).astype(mx.uint64), mx.zeros((trials,), dtype=mx.uint64), 0, 50))
    np.add.at(ours, np.array(picks), 1)
    mx.random.seed(5)
    np.add.at(vendored, np.array(sample_top_k_top_p(rows, 0.7, 30, 0.9)), 1)
    distance = 0.5 * float(np.abs(ours / trials - vendored / trials).sum())
    kept = int((vendored > 0).sum())
    return {"noise_matches_cuda_stream": noise_ok, "total_variation_vs_vendored": round(distance, 4),
            "tokens_drawn": kept, "passed": noise_ok and distance < 0.03 and kept >= 10}


def runner(function, *args):
    return function(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiny", action="store_true", help="small random model on the CPU (no GPU, no model files)")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=24)
    args = parser.parse_args()
    if args.tiny == bool(args.checkpoint):
        parser.error("give exactly one of --tiny or --checkpoint")
    import numpy as np
    import mlx.core as mx
    from drama3_mlx.generate import generate_frames
    if args.tiny:
        mx.set_default_device(mx.cpu)
        model = tiny_model()
        tolerance = 1e-4
    else:
        from drama3_mlx.prompt import build_prompt
        from epub_tts_narrator.renderer.engines import Drama3MLXEngine
        from check_drama3_silence_stop import MLX_TEXTS
        loaded = Drama3MLXEngine(args.checkpoint)._load()
        model = loaded.model
        tolerance = None
    rng = np.random.default_rng(0)
    if args.tiny:
        prompts = [random_prompt(model, rows, rng) for rows in (40, 23, 61, 33, 50, 28, 45, 37)[: args.rows]]
    else:
        # Random tokens drive the real model into chaotic states; real sentences do not.
        prompts = [build_prompt(MLX_TEXTS[i % len(MLX_TEXTS)], loaded.tokenizer, model.config) for i in range(args.rows)]
    frames = []
    for index, prompt in enumerate(prompts):
        mx.random.seed(index)
        sampled, _ = generate_frames(model, prompt, max_new_tokens=args.max_new + 4 * index, temperature=0.7, top_k=30, top_p=0.7)
        frames.append(sampled if len(sampled) else prompt[-1:])
    if args.tiny:
        report = {"rows": args.rows, "forced": check_forced(model, prompts, frames, tolerance),
                  "independence": check_independence(model, prompts, args.max_new)}
    else:
        # Exact row independence only holds where multi-row kernels are exact (the CPU).
        report = {"rows": args.rows, "forced": check_gpu_drift(model, prompts, frames)}
    report["sampler"] = check_sampler(20000)
    report["passed"] = all(part["passed"] for part in report.values() if isinstance(part, dict))
    print("MLX_BATCH_REPORT=" + json.dumps(report), flush=True)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
