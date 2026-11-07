"""Batched Drama 3 decoding on Fish's unmodified model (Runpod CUDA worker).

Fish's pinned decoder serves one request: its caches hold one row, its CUDA
graphs are captured for one row, and its attention takes one position shared
by every row. This backend keeps every model module and weight as Fish loads
them and decodes several independent requests per step:

* Prefill runs Fish's own ``backbone`` forward and ``tower.prefill`` for one
  request, with each cache attribute pointed at that request's row of a
  larger cache, so prompts (references, chapter history, text) are encoded
  exactly as Fish encodes them.
* Decode reuses Fish's linear-attention (Gated DeltaNet), MoE, norm,
  embedding and head modules unchanged. Only full attention in the backbone
  and the tower's time attention are recomputed here, because each row is at
  its own position: each row writes its own cache slot and attends through
  its own mask.
* Sampling follows Fish's ``sample_top_k_top_p`` and repetition-aware retry
  exactly, with per-row parameters, but draws its noise from a counter-based
  hash of (seed, step, draw) so a row's randomness never depends on which
  other requests share the batch.

Every torch and Fish import is function-local; only the CUDA worker's Fish
environment imports this module.
"""

from __future__ import annotations

# Counter-hash constants (splitmix64), written as signed int64.
_GOLDEN = 0x9E3779B97F4A7C15 - (1 << 64)
_MIX1 = 0xBF58476D1CE4E5B9 - (1 << 64)
_MIX2 = 0x94D049BB133111EB - (1 << 64)
SAMPLER = "counter-hash-v1"
DECODER_VERSION = 1


def _shift(x, bits):
    """Logical right shift of an int64 tensor."""
    return (x >> bits) & ((1 << (64 - bits)) - 1)


def _splitmix(x):
    x = x ^ _shift(x, 30)
    x = x * _MIX1
    x = x ^ _shift(x, 27)
    x = x * _MIX2
    return x ^ _shift(x, 31)


def sampling_noise(seed, step, draw, width):
    """Exponential(1) noise [rows, width] fixed by each row's (seed, step, draw)."""
    import torch
    key = _splitmix(seed * _GOLDEN + _splitmix(step * 64 + draw + 1))
    columns = torch.arange(width, device=seed.device, dtype=torch.int64)
    bits = _shift(_splitmix(key[:, None] + columns[None, :] * _GOLDEN), 40)
    uniform = (bits.to(torch.float32) + 0.5) * (1.0 / (1 << 24))
    return -torch.log(uniform)


def sample_rows(logits, temperature, top_k, top_p, noise):
    """Fish's ``sample_top_k_top_p`` with per-row parameters and supplied noise."""
    import torch
    from fish_speech.models.text2semantic.fish_omni.tower import MIN_TEMPERATURE
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.clamp(min=MIN_TEMPERATURE)
    else:
        temperature = max(temperature, MIN_TEMPERATURE)
    values, ids = torch.sort(logits.float() / temperature, descending=True, dim=-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= top_k)
    dropped[..., 0] = False
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    choice = (probabilities / noise).argmax(-1, keepdim=True)
    return ids.gather(-1, choice).squeeze(-1)


def _rope(x, cos, sin):
    """Fish's ``apply_neox_rope`` for x [rows, 1, heads, dim] with per-row angles."""
    import torch
    width = cos.shape[-1]
    rotary, passthrough = x[..., :width].float(), x[..., width:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    out = rotary * cos[:, None, None, :] + rotated * sin[:, None, None, :]
    return torch.cat((out.to(x.dtype), passthrough), dim=-1)


class BatchedDrama3Backend:
    """Per-row caches and one batched decode step; see the module docstring."""

    def __init__(self, model, *, rows, capacity=None, graphs=None):
        import torch
        from fish_speech.models.text2semantic.fish_omni.generate import (
            DRAMA3_GRAPH_KV_BUCKETS, Drama3DecodeGraphs)
        from fish_speech.models.text2semantic.fish_omni.tower import RAS_WINDOW_FRAMES

        if type(rows) is not int or rows < 1:
            raise ValueError("Batched Drama 3 needs at least one row")
        self.model, self.rows = model, rows
        backbone, tower = model.backbone, model.audio_decoder
        device = tower.device
        if any(p.device != device for p in backbone.parameters()):
            raise ValueError("Batched Drama 3 requires the backbone and tower on one device")
        self.capacity = capacity or Drama3DecodeGraphs.graph_capacity(model)
        self.eos_id = model.config.eos_token_id
        self.window = RAS_WINDOW_FRAMES
        # Fish's single-request decode state must never touch these caches.
        model.decode_graphs = None
        model.cache_capacity = 0
        backbone.setup_caches(max_batch_size=rows, max_seq_len=self.capacity)
        self.full = [layer.self_attn for layer in backbone.model.layers if layer.layer_type == "full_attention"]
        self.linear = [layer.linear_attn for layer in backbone.model.layers if layer.layer_type != "full_attention"]
        self.tower_attention = [layer.transformer_layer.attention
                                for layer in (*tower.shared_layers, *tower.residual_layers)]
        self.tower_caches = []
        for attention in self.tower_attention:
            config = attention.config
            shape = (rows, config.n_local_heads, self.capacity, config.head_dim)
            self.tower_caches.append((attention.wqkv.weight.new_zeros(shape),
                                      attention.wqkv.weight.new_zeros(shape)))
            attention.k_cache = attention.v_cache = None
            attention.static_kv_len = None
        width = 1 + tower.config.num_codebooks
        self.width = width
        long = dict(dtype=torch.long, device=device)
        self.hidden = torch.zeros((rows, model.config.text.hidden_size), dtype=backbone.dtype, device=device)
        self.next_hidden = torch.zeros_like(self.hidden)
        self.position = torch.zeros(rows, **long)
        self.frame = torch.zeros((rows, width), **long)
        self.sampled = torch.zeros((rows, width), **long)
        self.history = torch.full((rows, self.window), -1, **long)
        self.temperature = torch.ones((rows, 1), dtype=torch.float32, device=device)
        self.top_k = torch.ones((rows, 1), **long)
        self.top_p = torch.ones((rows, 1), dtype=torch.float32, device=device)
        self.seed = torch.zeros(rows, **long)
        self.step_index = torch.zeros(rows, **long)
        self.active = torch.zeros(rows, **long)
        self.finite = torch.ones(rows, dtype=torch.bool, device=device)
        self.row_index = torch.arange(rows, device=device)
        self.key_positions = torch.arange(self.capacity, device=device)
        self.positions = [0] * rows
        self.live = set()
        if tower.main_candidates is None or tower.main_candidates.device != device:
            eos = torch.tensor([tower.eos_id], device=device)
            band = torch.arange(tower.semantic_begin, tower.semantic_end + 1, device=device)
            tower.main_candidates = torch.cat((eos, band)).sort().values
        frequencies = backbone.inverse_frequencies.get(device)
        if frequencies is None:
            rotary = backbone.config.rotary_dim
            exponents = torch.arange(0, rotary, 2, dtype=torch.float32, device=device)
            frequencies = 1.0 / (backbone.config.rope_theta ** (exponents / rotary))
            backbone.inverse_frequencies[device] = frequencies
        self.frequencies = frequencies
        self.sampler = self._sample
        self.buckets = tuple(sorted({min(bucket, self.capacity) for bucket in DRAMA3_GRAPH_KV_BUCKETS}))
        self.graphs = {}
        self.use_graphs = device.type == "cuda" if graphs is None else graphs
        if self.use_graphs:
            self._capture()

    # Sampling -------------------------------------------------------------

    def _sample(self, logits, temperature, top_k, top_p, draw):
        noise = sampling_noise(self.seed, self.step_index, draw, logits.shape[-1])
        return sample_rows(logits, temperature, top_k, top_p, noise)

    # Backbone -------------------------------------------------------------

    def _attention(self, attention, x, positions, cos, sin, kv_len):
        import torch
        from fish_speech.models.text2semantic.fish_omni.layers import grouped_attention
        rows = x.shape[0]
        heads, kv_heads, dim = attention.num_heads, attention.num_kv_heads, attention.head_dim
        query, gate = attention.q_proj(x).view(rows, 1, heads, 2 * dim).chunk(2, dim=-1)
        gate = gate.reshape(rows, 1, heads * dim)
        query = attention.q_norm(query)
        key = attention.k_norm(attention.k_proj(x).view(rows, 1, kv_heads, dim))
        value = attention.v_proj(x).view(rows, 1, kv_heads, dim)
        query = _rope(query, cos, sin).transpose(1, 2)
        key = _rope(key, cos, sin).transpose(1, 2)
        value = value.transpose(1, 2)
        attention.k_cache[self.row_index, :, positions] = key[:, :, 0]
        attention.v_cache[self.row_index, :, positions] = value[:, :, 0]
        keys = attention.k_cache[:rows, :, :kv_len]
        values = attention.v_cache[:rows, :, :kv_len]
        mask = (self.key_positions[:kv_len][None, :] <= positions[:, None])[:, None, None, :]
        attended = grouped_attention(query, keys, values, mask, scale=dim ** -0.5)
        attended = attended.transpose(1, 2).reshape(rows, 1, heads * dim)
        return attention.o_proj(attended * torch.sigmoid(gate))

    def _backbone(self, embeddings, positions, kv_len):
        """Final-norm hidden [rows, H] for one new frame per row at its own position."""
        import torch
        from fish_speech.models.text2semantic.fish_omni.layers import LayerContext
        backbone = self.model.backbone
        angles = positions.float()[:, None] * self.frequencies[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos(), angles.sin()
        # Gated DeltaNet decodes each row on its own recurrent state; it reads
        # no positions, so Fish's module runs the whole batch unchanged.
        context = LayerContext(input_pos=positions, cos=cos, sin=sin, kv_len=kv_len,
                               continue_state=True, use_cache=True)
        hidden = embeddings[:, None].to(backbone.residual_dtype)
        for layer in backbone.model.layers:
            compute_dtype = layer.input_layernorm.weight.dtype
            normed = layer.input_layernorm(hidden).to(compute_dtype)
            if layer.layer_type == "full_attention":
                mixed = self._attention(layer.self_attn, normed, positions, cos, sin, kv_len)
            else:
                mixed = layer.linear_attn(normed, context)
            hidden = hidden + mixed
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden).to(compute_dtype))
        return backbone.model.norm(hidden).to(backbone.dtype)[:, 0]

    # Tower ----------------------------------------------------------------

    def _tower_attention(self, index, x, positions, kv_len):
        from fish_speech.models.text2semantic.fish_omni.layers import grouped_attention
        attention = self.tower_attention[index]
        config = attention.config
        rows = x.shape[0]
        q_width = config.n_head * config.head_dim
        kv_width = config.n_local_heads * config.head_dim
        q, gate, k, v = attention.wqkv(x).split((q_width, q_width, kv_width, kv_width), -1)
        q = attention.q_norm(q.reshape(rows, config.n_head, config.head_dim))
        k = attention.k_norm(k.reshape(rows, config.n_local_heads, config.head_dim))
        q = attention.rotate(q, positions)
        k = attention.rotate(k, positions)
        v = v.reshape(rows, config.n_local_heads, config.head_dim)
        k_cache, v_cache = self.tower_caches[index]
        k_cache[self.row_index, :, positions] = k
        v_cache[self.row_index, :, positions] = v
        mask = (self.key_positions[:kv_len][None, :] <= positions[:, None])[:, None, None, :]
        out = grouped_attention(q[:, :, None, :], k_cache[:, :, :kv_len], v_cache[:, :, :kv_len], mask)
        return attention.wo(out.reshape(rows, q_width) * gate.sigmoid())

    def _tower_layer(self, layer, index, hidden, codes, semantic, positions, kv_len, shared_input=None):
        """Fish's ``TowerLayer.forward`` with batched time attention."""
        import torch
        embedding = layer.semantic_embedding if layer.shared else layer.codebook_embedding
        codes = codes.clamp(0, embedding.num_embeddings - 1)
        e = torch.where(semantic[:, None], embedding(codes), 0.0)
        if shared_input is not None:
            e = e + shared_input
        mixed = layer.eh_proj(torch.cat((layer.enorm(e), layer.hnorm(hidden)), dim=-1))
        block = layer.transformer_layer
        dtype = block.attention.wqkv.weight.dtype
        residual = mixed.float() if dtype == torch.float16 else mixed
        residual = residual + self._tower_attention(
            index, block.attention_norm(residual).to(dtype), positions, kv_len)
        out = residual + block.feed_forward(block.ffn_norm(residual).to(dtype))
        return layer.final_layernorm(out).to(mixed.dtype)

    def _decode(self, logits, kv_len):
        """Fish's ``decode_step`` for every row: main token with retry, then codes."""
        import torch
        from fish_speech.models.text2semantic.fish_omni.tower import RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P
        tower = self.model.audio_decoder
        candidates = tower.main_candidates
        candidate_logits = logits[:, candidates]
        first = candidates[self.sampler(candidate_logits, self.temperature, self.top_k, self.top_p, 0)]
        retry = candidates[self.sampler(candidate_logits, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P, 1)]
        repeated = (self.history == first[:, None]).any(-1) & tower.is_semantic(first)
        main = torch.where(repeated, retry, first)
        c0 = (main - tower.semantic_begin).clamp(0, tower.config.vocab_size - 1)
        semantic = tower.is_semantic(main)
        unseen = tower.embedding_sum(self.frame, 1 + tower.config.vq_backbone_codebooks,
                                     tower.config.num_codebooks)
        unseen = torch.where(tower.is_semantic(self.frame[:, 0])[:, None], unseen, 0.0)
        hidden = self.hidden
        for index, layer in enumerate(tower.shared_layers):
            hidden = self._tower_layer(layer, index, hidden, c0, semantic, self.position, kv_len,
                                       unseen if index == 0 else None)
        codes = [c0]
        offset = len(tower.shared_layers)
        for index, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
            hidden = self._tower_layer(layer, offset + index, hidden, codes[-1], semantic,
                                       self.position, kv_len)
            codes.append(self.sampler(head(hidden), self.temperature, self.top_k, self.top_p, 2 + index))
        return torch.stack((main, *codes), dim=-1)

    def _run(self, kv_len):
        """One decode step for every row (the body of each captured graph)."""
        import torch
        backbone, tower = self.model.backbone, self.model.audio_decoder
        logits = backbone.compute_logits(self.hidden).to(tower.device)
        self.finite.copy_(torch.isfinite(logits).all(-1))
        sampled = self._decode(logits, kv_len)
        self.sampled.copy_(sampled)
        embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
        self.next_hidden.copy_(self._backbone(embeddings, self.position + 1, kv_len))

    def _capture(self):
        import torch
        pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode():
            for kv_len in self.buckets:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._run(kv_len)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self._run(kv_len)
                self.graphs[kv_len] = graph
        # Capture wrote placeholder values; every slot is prefilled before use.
        self.active.zero_()
        self.position.zero_()

    # Scheduler interface --------------------------------------------------

    def _slot_views(self, slot):
        """Point every Fish cache attribute at this slot's row; returns a restore list."""
        saved = []
        for attention in self.full:
            saved.append((attention, "k_cache", attention.k_cache))
            saved.append((attention, "v_cache", attention.v_cache))
            attention.k_cache = attention.k_cache[slot:slot + 1]
            attention.v_cache = attention.v_cache[slot:slot + 1]
        for attention in self.linear:
            saved.append((attention, "conv_state", attention.conv_state))
            saved.append((attention, "recurrent_state", attention.recurrent_state))
            attention.conv_state = attention.conv_state[slot:slot + 1]
            attention.recurrent_state = attention.recurrent_state[slot:slot + 1]
        for attention, (k_cache, v_cache) in zip(self.tower_attention, self.tower_caches):
            saved.append((attention, "k_cache", attention.k_cache))
            saved.append((attention, "v_cache", attention.v_cache))
            attention.k_cache = k_cache[slot:slot + 1]
            attention.v_cache = v_cache[slot:slot + 1]
        return saved

    def prefill(self, slot, request):
        """Fish's prefill for one request, into this slot's row of every cache."""
        import torch
        from fish_speech.models.text2semantic.fish_omni.generate import DRAMA3_PREFILL_CHUNK_SIZE
        backbone, tower = self.model.backbone, self.model.audio_decoder
        prompt = request.prompt
        rows = len(prompt)
        if not 0 < rows < self.capacity or request.max_new_tokens > self.capacity - rows:
            raise ValueError("The Drama 3 request does not fit the batched decode capacity")
        saved = self._slot_views(slot)
        try:
            with torch.inference_mode():
                frames = prompt.to(tower.device)
                positions = torch.arange(rows, device=tower.device)
                hidden = None
                for start in range(0, rows, DRAMA3_PREFILL_CHUNK_SIZE):
                    end = min(start + DRAMA3_PREFILL_CHUNK_SIZE, rows)
                    chunk = frames[start:end]
                    embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
                    hidden = backbone(inputs_embeds=embeddings[None], input_pos=positions[start:end])
                    hidden = hidden[0].to(tower.device)
                    paired_end = min(end, rows - 1)
                    if paired_end > start:
                        tower.prefill(hidden[:paired_end - start], positions[start:paired_end],
                                      frames[start:paired_end], frames[start + 1:paired_end + 1])
                window = frames[-self.window:, 0]
                self.hidden[slot] = hidden[-1]
                self.position[slot] = rows - 1
                self.frame[slot] = frames[-1]
                self.history[slot] = -1
                self.history[slot, self.window - len(window):] = window
                self.temperature[slot] = float(request.temperature)
                self.top_k[slot] = int(request.top_k)
                self.top_p[slot] = float(request.top_p)
                self.seed[slot] = int(request.seed)
                self.step_index[slot] = 0
                self.active[slot] = 1
        finally:
            for module, name, value in saved:
                setattr(module, name, value)
        self.positions[slot] = rows - 1
        self.live.add(slot)

    def release(self, slot):
        import torch
        self.live.discard(slot)
        self.positions[slot] = 0
        with torch.inference_mode():
            self.active[slot] = 0
            self.position[slot] = 0

    def step(self, slots):
        import torch
        live = [slot for slot in slots if slot in self.live]
        if live != list(slots):
            raise RuntimeError("Batched Drama 3 was asked to step an empty slot")
        needed = max(self.positions[slot] for slot in slots) + 2
        kv_len = next(bucket for bucket in self.buckets if bucket >= needed)
        with torch.inference_mode():
            if self.use_graphs:
                self.graphs[kv_len].replay()
            else:
                self._run(kv_len)
            sampled = self.sampled.clone()
            status = torch.cat((sampled.flatten(), self.finite.long()))
            self.hidden.copy_(self.next_hidden)
            self.frame.copy_(sampled)
            self.history.copy_(torch.cat((self.history[:, 1:], sampled[:, :1]), dim=-1))
            # Idle rows stay at position 0 so they can never outgrow the caches.
            self.position.add_(1).mul_(self.active)
            self.step_index.add_(1)
            values = status.tolist()
        width, rows = self.width, self.rows
        for slot in slots:
            self.positions[slot] += 1
        return [(values[slot * width:(slot + 1) * width], bool(values[rows * width + slot]))
                for slot in slots]
