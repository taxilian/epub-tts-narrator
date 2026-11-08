"""Batched Drama 3 decoding on the vendored MLX model (Apple silicon).

The vendored ``drama3_mlx`` loop decodes one request: its caches hold one
sequence and its codebook tower takes one sequence. This backend keeps every
vendored module and weight unchanged and decodes several independent
requests per step, for ``drama3_batch_scheduler.Drama3BatchScheduler``:

* Prefill runs the vendored prefill for one request alone (backbone chunks
  and ``tower.forced``), exactly as ``generate_frames`` does, into that
  request's own caches, which are then appended to the batch as one more row
  (mlx-lm's ``BatchKVCache`` / ``ArraysCache``, which keep each row at its own
  position behind left padding).
* Decode runs the vendored backbone unchanged on every row at once: its full
  attention takes per-row offsets and masks from ``BatchKVCache``, and its
  linear attention keeps per-row state. Only the tower's time attention is
  recomputed here (``_tower_attention``), because the vendored one reshapes
  for a single sequence; it uses the same weights, norms and rotary
  embedding. A finished row is filtered out of every cache.
* Sampling follows the vendored ``sample_top_k_top_p`` and repetition-aware
  retry exactly, with per-row parameters, but draws its noise from the same
  counter-based hash of (seed, step, draw) as the CUDA backend
  (``drama3_batch``), so a row's randomness never depends on which other
  requests share the batch.

A prompt's shared start (a voice's references, then the chapter's pinned
history turn) is the same for every take of that voice, so its prefill is
cached (``_PrefixCache``): the state after a 128-row prefill chunk boundary
inside that start is saved once, keyed by a hash of the exact rows, and a
later prompt beginning with the same rows resumes from it. The vendored
prefill runs in 128-row chunks from row 0, so resuming at a chunk boundary
runs the same computations in the same order: only the last cached chunk's
tower pairs are redone, because they read the prompt's next row. Snapshots
are copies; mlx-lm caches write into their buffers in place.

MLX streams belong to the thread that created them, so every MLX operation
runs through ``run`` (the engine passes its one persistent MLX thread). Every
MLX import is function-local.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

SAMPLER = "counter-hash-v2"
DECODER_VERSION = 1
MAX_BATCH_ROWS = 16
# Prefill snapshots kept for reuse (least recently used evicted first).
PREFIX_CACHE_ENTRIES = 32
PREFIX_CACHE_BYTES = 4 << 30
# splitmix64 constants (the CUDA backend writes the same bits as signed int64).
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB


def _u64(value):
    # mx.array cannot take a Python int above 2**63 as uint64; NumPy can.
    import numpy as np
    import mlx.core as mx
    return mx.array(np.array(value, dtype=np.uint64))


def _splitmix(x):
    x = x ^ (x >> _u64(30))
    x = x * _u64(_MIX1)
    x = x ^ (x >> _u64(27))
    x = x * _u64(_MIX2)
    return x ^ (x >> _u64(31))


def sampling_noise(seed, step, draw, width):
    """Exponential(1) noise [rows, width], strictly positive, fixed by each row's (seed, step, draw).

    Bit-for-bit the CUDA backend's stream: the top 23 bits of each hash keep
    ``bits + 0.5`` exact in float32, so the uniform never rounds to 0 or 1.
    """
    import mlx.core as mx
    seed, step = seed.astype(mx.uint64), step.astype(mx.uint64)
    key = _splitmix(seed * _u64(_GOLDEN) + _splitmix(step * _u64(64) + _u64(draw + 1)))
    columns = mx.arange(width).astype(mx.uint64)
    hashes = _splitmix(key[:, None] + columns[None, :] * _u64(_GOLDEN))
    uniform = ((hashes >> _u64(41)).astype(mx.float32) + 0.5) * (1.0 / (1 << 23))
    return -mx.log(uniform)


def sample_rows(logits, temperature, top_k, top_p, noise):
    """The vendored ``sample_top_k_top_p`` with per-row parameters [rows] and supplied noise.

    Drawing ``argmax(p / E)`` with ``E ~ Exponential(1)`` picks token i with
    probability p_i, like the vendored categorical draw; the crossing token is
    dropped and the top token always kept, as there.
    """
    import mlx.core as mx
    from drama3_mlx.sampling import MIN_TEMPERATURE
    scaled = logits.astype(mx.float32) / mx.maximum(temperature, MIN_TEMPERATURE)[:, None]
    order = mx.argsort(-scaled, axis=-1)
    values = mx.take_along_axis(scaled, order, axis=-1)
    ranks = mx.arange(values.shape[-1])[None, :]
    cumulative = mx.cumsum(mx.softmax(values, axis=-1), axis=-1)
    dropped = ((cumulative > top_p[:, None]) | (ranks >= top_k[:, None])) & (ranks > 0)
    probabilities = mx.softmax(mx.where(dropped, -mx.inf, values), axis=-1)
    choice = mx.argmax(probabilities / noise, axis=-1)
    return mx.take_along_axis(order, choice[:, None], axis=-1).squeeze(-1)


def _tower_attention(attention, x, cache):
    """The vendored ``TowerAttention`` for one new frame per row: x [rows, dim]."""
    import mlx.core as mx
    cfg = attention.config
    rows = x.shape[0]
    q_width, kv_width = cfg.n_head * cfg.head_dim, cfg.n_local_heads * cfg.head_dim
    q, gate, k, v = mx.split(attention.wqkv(x), [q_width, 2 * q_width, 2 * q_width + kv_width], axis=-1)
    q = attention.q_norm(q.reshape(rows, 1, cfg.n_head, cfg.head_dim))
    k = attention.k_norm(k.reshape(rows, 1, cfg.n_local_heads, cfg.head_dim))
    v = v.reshape(rows, 1, cfg.n_local_heads, cfg.head_dim)
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    q = attention.rope(q, offset=cache.offset)
    k = attention.rope(k, offset=cache.offset)
    mask = cache.make_mask(1)
    k, v = cache.update_and_fetch(k, v)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=cfg.head_dim ** -0.5, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(rows, q_width)
    return attention.wo(out * mx.sigmoid(gate))


def _tower_layer(layer, hidden, codes, semantic, cache, shared_input=None):
    """The vendored ``TowerLayer`` with the batched time attention."""
    import mlx.core as mx
    embedding = layer.semantic_embedding if layer.shared else layer.codebook_embedding
    e = embedding(mx.clip(codes, 0, layer.vocab_size - 1))
    e = mx.where(semantic[:, None], e, mx.zeros_like(e))
    if shared_input is not None:
        e = e + shared_input
    x = layer.eh_proj(mx.concatenate([layer.enorm(e), layer.hnorm(hidden)], axis=-1))
    block = layer.transformer_layer
    x = x + _tower_attention(block.attention, block.attention_norm(x), cache)
    x = x + block.feed_forward(block.ffn_norm(x))
    return layer.final_layernorm(x)


def _copy_cache(cache):
    """An independent copy of a single-request cache (mlx-lm caches update their buffers in place)."""
    from mlx_lm.models.cache import ArraysCache, KVCache
    if isinstance(cache, KVCache):
        copy = KVCache()
        if cache.keys is not None:
            # Exactly the filled rows: the copy's next update allocates a new buffer.
            copy.keys = cache.keys[..., : cache.offset, :] + 0
            copy.values = cache.values[..., : cache.offset, :] + 0
            copy.offset = cache.offset
        return copy
    if isinstance(cache, ArraysCache):
        copy = ArraysCache(len(cache.cache))
        copy.cache = [None if state is None else state + 0 for state in cache.cache]
        return copy
    raise TypeError(f"Unsupported Drama 3 cache {type(cache).__name__}")


def _cache_bytes(caches):
    from mlx_lm.models.cache import ArraysCache
    total = 0
    for cache in caches:
        if isinstance(cache, ArraysCache):
            total += sum(state.nbytes for state in cache.cache if state is not None)
        elif cache.keys is not None:
            total += cache.keys.nbytes + cache.values.nbytes
    return total


class _PrefixCache:
    """Prefill state after an exact prompt prefix, least recently used evicted first."""

    def __init__(self, entries=PREFIX_CACHE_ENTRIES, max_bytes=PREFIX_CACHE_BYTES):
        self._entries, self._max_entries, self._max_bytes = OrderedDict(), entries, max_bytes
        self.bytes = 0
        self.hits = self.misses = self.rows_reused = self.stored = 0

    @staticmethod
    def key(prompt, rows):
        return hashlib.sha256(prompt[:rows].astype("int64").tobytes()).hexdigest()

    def find(self, prompt, chunk):
        """(rows, entry) for the longest cached chunk-aligned prefix that leaves a row to decode, else None."""
        for rows in range((len(prompt) - 1) // chunk * chunk, 0, -chunk):
            key = self.key(prompt, rows)
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                self.rows_reused += rows
                return rows, self._entries[key]
        self.misses += 1
        return None

    def __contains__(self, key):
        return key in self._entries

    def store(self, key, entry):
        entry["bytes"] = (_cache_bytes(entry["caches"]) + _cache_bytes(entry["tower_caches"])
                          + entry["hidden"].nbytes)
        self._entries[key] = entry
        self.bytes += entry["bytes"]
        self.stored += 1
        while self._entries and (len(self._entries) > self._max_entries or self.bytes > self._max_bytes):
            _key, evicted = self._entries.popitem(last=False)
            self.bytes -= evicted["bytes"]

    def stats(self):
        return {"entries": len(self._entries), "bytes": self.bytes, "hits": self.hits, "misses": self.misses,
                "rows_reused": self.rows_reused, "stored": self.stored}


def prefix_ends(tokenizer, config, text, prompt_texts, prompt_codes, history):
    """Prompt rows before each history turn: [after the references, after turn 1, ...].

    Without history, the one shared part ends after the references.

    These are the parts of ``build_prompt``'s layout that later takes of the
    same voice repeat. A wrong value only costs a cache miss: entries are keyed
    by the exact rows.
    """
    from drama3_mlx.prompt import GENERATION_START, MESSAGE_END, build_prompt, with_speaker_tag
    current = sum(len(tokenizer.encode(segment).ids) for segment in (
        "<|im_start|>user\n", with_speaker_tag(text), MESSAGE_END, GENERATION_START))
    # The end of the references is shared even by a voice's first take (no history).
    return [len(build_prompt(text, tokenizer, config, prompt_texts, prompt_codes, history[:turns])) - current
            for turns in range(max(1, len(history)))]


def _batched(cache):
    """One request's prefilled cache as a one-row batch cache."""
    from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache
    if isinstance(cache, KVCache):
        return BatchKVCache.merge([cache])
    if isinstance(cache, ArraysCache):
        return ArraysCache.merge([cache])
    raise TypeError(f"Unsupported Drama 3 cache {type(cache).__name__}")


class BatchedMLXBackend:
    """Per-row caches and sampling state behind ``Drama3BatchScheduler``'s backend contract."""

    def __init__(self, model, *, rows, run=None):
        """``model`` is the vendored ``Drama3Model`` (``Drama3.model``); ``run(fn)`` executes on the MLX thread."""
        if type(rows) is not int or not 1 <= rows <= MAX_BATCH_ROWS:
            raise ValueError(f"rows must be an integer from 1 through {MAX_BATCH_ROWS}")
        self.model, self.rows = model, rows
        self.eos_id = model.config.eos_token_id
        self._run = run or (lambda function, *args: function(*args))
        self.order = []          # scheduler slots, in batch-row order
        self._state = None
        self.prefixes = _PrefixCache()

    # Scheduler contract -------------------------------------------------

    def prefill(self, slot, request):
        if slot in self.order or not 0 <= slot < self.rows:
            raise RuntimeError("Batched Drama 3 slot is already in use or out of range")
        self._run(self._prefill, slot, request)

    def release(self, slot):
        if slot in self.order:
            self._run(self._release, slot)

    def step(self, slots):
        return self._run(self._step, list(slots), None)

    def forced_step(self, slots, frames):
        """Teacher-forced step for equivalence checks: each row takes ``frames[slot]``.

        Returns ``{slot: (main logits [V], head logits [codebooks - 1, vocab])}``.
        """
        return self._run(self._step, list(slots), frames)

    # MLX-thread work ----------------------------------------------------

    def _prefill(self, slot, request):
        import numpy as np
        import mlx.core as mx
        from drama3_mlx.generate import PREFILL_CHUNK_SIZE, run_backbone
        from drama3_mlx.sampling import RAS_WINDOW_FRAMES
        model, config, tower = self.model, self.model.config, self.model.tower
        prompt = np.asarray(request.prompt)
        prompt_rows = len(prompt)
        if (prompt.ndim != 2 or prompt.shape[1] != 1 + config.num_codebooks or prompt_rows < 1
                or prompt_rows + request.max_new_tokens > config.text["max_position_embeddings"]):
            raise ValueError("The Drama 3 request does not fit the batched decoder")
        chunk = PREFILL_CHUNK_SIZE
        rows = mx.array(prompt.astype(np.int32))
        # Chunk boundaries inside the prompt's shared start, worth keeping for later takes.
        keep = {end // chunk * chunk for end in getattr(request, "prefix_hints", ()) or ()
                if chunk <= end // chunk * chunk <= prompt_rows - 1}
        found = self.prefixes.find(prompt, chunk) if keep or len(self.prefixes._entries) else None
        if found is None:
            first = 0
            caches = model.backbone.make_cache()
            tower_caches = tower.make_caches()
        else:
            first, entry = found
            caches = [_copy_cache(c) for c in entry["caches"]]
            tower_caches = [_copy_cache(c) for c in entry["tower_caches"]]
            # The cached chunk's tower pairs read the row after it: this prompt's.
            start = first - chunk
            tower.forced(entry["hidden"], rows[start:first], rows[start + 1: first + 1], tower_caches)
            hidden = entry["hidden"]
            mx.eval([c.state for c in tower_caches])
        for start in range(first, prompt_rows, chunk):
            end = min(start + chunk, prompt_rows)
            hidden = run_backbone(model, rows[start:end], caches)
            if end in keep:
                key = self.prefixes.key(prompt, end)
                if key not in self.prefixes:
                    mx.eval(hidden, [c.state for c in caches], [c.state for c in tower_caches])
                    self.prefixes.store(key, {"caches": [_copy_cache(c) for c in caches],
                                              "tower_caches": [_copy_cache(c) for c in tower_caches],
                                              "hidden": hidden + 0})
            paired_end = min(end, prompt_rows - 1)
            if paired_end > start:
                tower.forced(hidden[: paired_end - start], rows[start:paired_end],
                             rows[start + 1: paired_end + 1], tower_caches)
            mx.eval(hidden, [c.state for c in caches], [c.state for c in tower_caches])
        window = rows[-RAS_WINDOW_FRAMES:, 0]
        if window.shape[0] < RAS_WINDOW_FRAMES:  # -1 never equals a sampled token
            window = mx.concatenate([mx.full((RAS_WINDOW_FRAMES - window.shape[0],), -1, dtype=window.dtype), window])
        row = {
            "caches": [_batched(c) for c in caches],
            "tower_caches": [_batched(c) for c in tower_caches],
            "hidden": hidden[-1:], "frame": rows[-1:], "history": window[None, :],
            "temperature": mx.array([float(request.temperature)], dtype=mx.float32),
            "top_k": mx.array([int(request.top_k)], dtype=mx.int32),
            "top_p": mx.array([float(request.top_p)], dtype=mx.float32),
            "seed": mx.array(np.array([int(request.seed) % (1 << 64)], dtype=np.uint64)),
            "step": mx.array([0], dtype=mx.uint64),
        }
        if self._state is None:
            self._state = row
        else:
            state = self._state
            for name in ("caches", "tower_caches"):
                for batch, single in zip(state[name], row[name]):
                    batch.extend(single)
            for name in ("hidden", "frame", "history", "temperature", "top_k", "top_p", "seed", "step"):
                state[name] = mx.concatenate([state[name], row[name]])
        mx.eval(*[self._state[name] for name in ("hidden", "frame", "history")])
        self.order.append(slot)

    def _release(self, slot):
        import mlx.core as mx
        index = self.order.index(slot)
        self.order.pop(index)
        if not self.order:
            self._state = None
            return
        keep = mx.array([i for i in range(len(self.order) + 1) if i != index], dtype=mx.int32)
        state = self._state
        for name in ("caches", "tower_caches"):
            for cache in state[name]:
                cache.filter(keep)
        for name in ("hidden", "frame", "history", "temperature", "top_k", "top_p", "seed", "step"):
            state[name] = state[name][keep]

    def _step(self, slots, forced):
        import mlx.core as mx
        from drama3_mlx.sampling import RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P
        if sorted(slots) != sorted(self.order) or len(set(slots)) != len(slots):
            raise RuntimeError("Batched Drama 3 was asked to step a different set of rows")
        model, tower, state = self.model, self.model.tower, self._state
        backbone, config = model.backbone, model.config
        batch = len(self.order)
        hidden, frame, history = state["hidden"], state["frame"], state["history"]
        seed, step = state["seed"], state["step"]
        logits = backbone.lm_head(hidden)
        finite = mx.isfinite(logits).all(axis=-1)
        if forced is not None:
            given = mx.array([forced[slot] for slot in self.order], dtype=mx.int32)
            main, codes = given[:, 0], [given[:, 1 + i] for i in range(config.num_codebooks)]
        else:
            candidates = tower._main_candidates
            scores = logits[:, candidates]
            width = scores.shape[-1]
            first = candidates[sample_rows(scores, state["temperature"], state["top_k"], state["top_p"],
                                           sampling_noise(seed, step, 0, width))]
            retry = candidates[sample_rows(scores, mx.full((batch,), RAS_TEMPERATURE, dtype=mx.float32),
                                           mx.full((batch,), RAS_TOP_K, dtype=mx.int32),
                                           mx.full((batch,), RAS_TOP_P, dtype=mx.float32),
                                           sampling_noise(seed, step, 1, width))]
            repeated = (history == first[:, None]).any(axis=-1) & tower.is_semantic(first)
            main = mx.where(repeated, retry, first)
            codes = None
        semantic = tower.is_semantic(main)
        c0 = mx.clip(main - tower.semantic_begin, 0, tower.cfg.vocab_size - 1)
        unseen = tower.embedding_sum(frame, 1 + tower.cfg.vq_backbone_codebooks, tower.cfg.num_codebooks)
        unseen = mx.where(tower.is_semantic(frame[:, 0])[:, None], unseen, mx.zeros_like(unseen))
        h = hidden
        caches = state["tower_caches"]
        for index, layer in enumerate(tower.shared_layers):
            h = _tower_layer(layer, h, c0, semantic, caches[index], unseen if index == 0 else None)
        sampled, head_logits = [c0], []
        shared = len(tower.shared_layers)
        for index, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
            previous = sampled[-1] if codes is None else codes[index]
            h = _tower_layer(layer, h, previous, semantic, caches[shared + index])
            scores = head(h)
            if codes is None:
                sampled.append(sample_rows(scores, state["temperature"], state["top_k"], state["top_p"],
                                           sampling_noise(seed, step, 2 + index, scores.shape[-1])))
            else:
                head_logits.append(scores)
        new_frame = (mx.stack([main, *sampled], axis=-1) if codes is None
                     else mx.stack([main, *codes], axis=-1)).astype(frame.dtype)
        state["history"] = mx.concatenate([history[:, 1:], main[:, None].astype(history.dtype)], axis=1)
        embeddings = tower.embed_backbone(new_frame, backbone.model.embed_tokens(new_frame[:, 0]))
        state["hidden"] = backbone.model(None, cache=state["caches"], input_embeddings=embeddings[:, None, :])[:, -1]
        state["frame"], state["step"] = new_frame, step + _u64(1)
        mx.async_eval(state["hidden"], state["history"], state["step"],
                      [c.state for c in state["caches"]], [c.state for c in caches])
        if forced is not None:
            stacked = mx.stack(head_logits, axis=1)
            mx.eval(logits, stacked)
            return {slot: (logits[i], stacked[i]) for i, slot in enumerate(self.order)}
        values = mx.concatenate([new_frame.reshape(-1).astype(mx.int32), finite.astype(mx.int32)]).tolist()
        width = config.num_codebooks + 1
        by_slot = {slot: (values[i * width:(i + 1) * width], bool(values[batch * width + i]))
                   for i, slot in enumerate(self.order)}
        return [by_slot[slot] for slot in slots]
