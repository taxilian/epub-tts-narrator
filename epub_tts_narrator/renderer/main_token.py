"""The open main-token draw for narration (``[narration_sampling] open_main_token``).

Drama 3 draws each frame's main token (the stop token or a semantic token)
before its codebook values. The vendored sampler draws it with the request's
temperature, top-p and top-k, and drops the token whose cumulative probability
crosses top_p. At narration's cold settings (0.7 / 0.7 / 30) that cuts the
stop token and most speech tokens out during a pause, so a take can stay
silent until the silence watch stops it. In a controlled test on 40 narration
passages that had gone silent (5 voices, 2 seeds, ``dist-runs/silence-sampling-ab``),
the vendored draw stalled on 27 of 80 takes and this one on none, with the
same transcript coverage.

``RULE`` draws the main token at temperature 1.0 and top-p 0.95, keeping the
token that crosses top_p, with no top-k cap and tokens under probability 0.001
dropped (Fish's reference epsilon); the highest-ranked token is always kept.
The repetition redraw uses the same rule at its own temperature. Codebook
values keep the request's sampling, exactly as the vendored step draws them.

A take records the rule it was drawn with (``engine_metadata["main_token"]``).
The rule is not part of a request's identity, so takes drawn before it keep
their requests and nothing is generated again because of it; it applies to
takes generated from now on. Only the single-request MLX loop draws it; other
decoders (batched MLX/CUDA, Fish's CUDA library, MLX-Audio) draw as before and
record nothing.
"""

from __future__ import annotations

RULE = "open-v1"
TEMPERATURE = 1.0
TOP_P = 0.95
EPSILON = 1e-3


def sample_open(logits, temperature, top_p=TOP_P, epsilon=EPSILON):
    """One id per row of ``logits`` [N, V]: nucleus keeping the crossing token, no rank cap."""
    import mlx.core as mx
    scaled = logits.astype(mx.float32) / temperature
    order = mx.argsort(-scaled, axis=-1)
    values = mx.take_along_axis(scaled, order, axis=-1)
    probs = mx.softmax(values, axis=-1)
    before = mx.cumsum(probs, axis=-1) - probs
    ranks = mx.arange(values.shape[-1])
    dropped = ((before >= top_p) | (probs < epsilon)) & (ranks > 0)
    choice = mx.random.categorical(mx.where(dropped, -mx.inf, values), axis=-1)
    return mx.take_along_axis(order, choice[..., None], axis=-1).squeeze(-1)


def decode_step(tower, logits, hidden, frame, history, caches, *, temperature, top_k, top_p):
    """The vendored ``Tower.decode_step`` with the open main-token draw.

    The same operations in the same order: main draw, repetition redraw,
    shared layers, then one codebook draw per residual layer at the request's
    sampling.
    """
    import mlx.core as mx
    from drama3_mlx.sampling import RAS_TEMPERATURE, RAS_WINDOW_FRAMES, sample_top_k_top_p
    candidates = tower._main_candidates
    candidate_logits = logits[:, candidates]
    first = candidates[sample_open(candidate_logits, TEMPERATURE)]
    retry = candidates[sample_open(candidate_logits, RAS_TEMPERATURE)]
    window = history[-RAS_WINDOW_FRAMES:]
    repeated = (window[None, :] == first[:, None]).any(axis=-1) & tower.is_semantic(first)
    main = mx.where(repeated, retry, first)
    c0 = mx.clip(main - tower.semantic_begin, 0, tower.cfg.vocab_size - 1)
    semantic = tower.is_semantic(main)
    hidden = tower.run_shared(hidden, c0, semantic, frame, caches)
    codes = [c0]
    shared = len(tower.shared_layers)
    for index, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
        hidden = layer(hidden, codes[-1], semantic, caches[shared + index])
        codes.append(sample_top_k_top_p(head(hidden), temperature, top_k, top_p))
    return mx.stack([main, *codes], axis=-1)


def requested():
    """The main-token rule this request asks for (``request_sampling``), or None."""
    from .request_sampling import current_sampling
    rule = (current_sampling() or {}).get("main_token")
    if rule not in (None, RULE):
        raise ValueError(f"Unknown main-token rule: {rule!r}")
    return rule
