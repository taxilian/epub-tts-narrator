"""Scoped sampling experiment for the pinned Drama 3 MLX runtime.

This proxy separates the semantic main-token cap from the residual-codebook
cap without modifying a loaded model or the installed package. It is opt-in;
the production engine does not enable it automatically.

SGLang-Omni's Drama 3 cookbook documents an uncapped main draw and residual
top-k <= 30. Its sampling.py at revision
4809648d486bdff88dee5199b3e448af3c250f00 retains fixed RAS settings (1.0/30/.95)
and always makes both draws. See:
https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html#defaults-parameters
https://github.com/sgl-project/sglang-omni/blob/4809648d486bdff88dee5199b3e448af3c250f00/sglang_omni/models/fishaudio_drama3/sampling.py

For a cap-only ablation, this module deliberately preserves the installed MLX
top-p policy: it excludes the crossing token for all draws. SGLang retains that
token for semantic draws, which is a separate experimental variable. Setting
both caps to 30 with request top_k=30 preserves the installed decode operation
order and random draws, including the RAS draw even when it is unused.
"""

from __future__ import annotations

from typing import Any


def _runtime():
    """Import MLX only when the proxy actually decodes a frame."""
    import mlx.core as mx
    from drama3_mlx import sampling
    return mx, sampling


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class _TowerProxy:
    def __init__(self, tower, main_top_k: int | None, codebook_top_k: int):
        self._tower = tower
        self.main_top_k = main_top_k
        self.codebook_top_k = codebook_top_k

    def __getattr__(self, name):
        return getattr(self._tower, name)

    def decode_step(
        self, logits, hidden, frame, history, caches, *, temperature, top_k, top_p,
    ):
        """Pinned MLX decode sequence with independently bounded head draws."""
        _positive_integer(top_k, "top_k")
        mx, sampling = _runtime()
        sample = sampling.sample_top_k_top_p
        candidates = self._main_candidates
        candidate_logits = logits[:, candidates]
        main_cap = len(candidates) if self.main_top_k is None else self.main_top_k
        codebook_cap = min(top_k, self.codebook_top_k, 30)
        first = candidates[sample(candidate_logits, temperature, main_cap, top_p)]
        retry = candidates[sample(
            candidate_logits, sampling.RAS_TEMPERATURE,
            sampling.RAS_TOP_K, sampling.RAS_TOP_P,
        )]
        window = history[-sampling.RAS_WINDOW_FRAMES:]
        repeated = (window[None, :] == first[:, None]).any(axis=-1)
        repeated = repeated & self.is_semantic(first)
        main = mx.where(repeated, retry, first)
        c0 = mx.clip(main - self.semantic_begin, 0, self.cfg.vocab_size - 1)
        semantic = self.is_semantic(main)
        hidden = self.run_shared(hidden, c0, semantic, frame, caches)
        codes = [c0]
        shared = len(self.shared_layers)
        for index, (layer, head) in enumerate(zip(self.residual_layers, self.codebook_heads)):
            hidden = layer(hidden, codes[-1], semantic, caches[shared + index])
            codes.append(sample(head(hidden), temperature, codebook_cap, top_p))
        return mx.stack([main, *codes], axis=-1)


class _ModelProxy:
    def __init__(self, model, main_top_k: int | None, codebook_top_k: int):
        self._model = model
        self.tower = _TowerProxy(model.tower, main_top_k, codebook_top_k)

    def __getattr__(self, name):
        return getattr(self._model, name)


def wrap_model(model, *, main_top_k: int | None = None, codebook_top_k: int = 30):
    """Return a generate_frames-compatible view; leave model and package untouched.

    ``main_top_k=None`` retains every semantic candidate before applying top-p.
    Residual heads use the smallest of the request top_k, codebook_top_k and 30.
    RAS remains fixed at temperature=1, top_k=30 and top_p=.95.
    """
    if main_top_k is not None:
        _positive_integer(main_top_k, "main_top_k")
    _positive_integer(codebook_top_k, "codebook_top_k")
    return _ModelProxy(model, main_top_k, codebook_top_k)
