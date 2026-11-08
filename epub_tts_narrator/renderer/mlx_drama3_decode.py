"""Drama 3 MLX generation that can end a take once it has gone silent.

``generate_frames_watched`` is the pinned vendored
``drama3_mlx.generate.generate_frames`` (drama3-mlx 0.1.1, upstream 3b76bf8)
reproduced step for step, plus one call: after each kept frame,
``watch.observe(codes)`` receives that frame's codebook values (without the
main token), and a True result ends the take with finish reason
``"silence"``. The watch only reads frames, so a take it does not stop has the
same frames as the vendored loop for the same seed; a stopped take is an exact
prefix. (The stop token's rank is no usable shortcut for finding silence:
measured on this loop it overlaps too much between speech and silence.) ``generate_segment`` is ``Drama3.generate(single_chunk=True,
attempts=1)`` on this loop. ``scripts/check_drama3_silence_stop.py --mlx``
checks both against the vendored runtime; rerun it whenever the vendored
snapshot changes. The vendored tree itself is never edited.
"""

from __future__ import annotations

import time


def generate_frames_watched(model, prompt, *, max_new_tokens, temperature, top_k, top_p, watch=None):
    """``generate_frames`` with an optional per-frame watch: (frames, finish reason)."""
    import numpy as np
    import mlx.core as mx
    from drama3_mlx.generate import PREFILL_CHUNK_SIZE, run_backbone
    from drama3_mlx.prompt import InvalidInputError
    from drama3_mlx.sampling import RAS_WINDOW_FRAMES

    if watch is not None and not callable(getattr(watch, "observe", None)):
        raise ValueError("watch must have an observe(codes) method")
    if top_k < 1:
        raise InvalidInputError(f"top_k must be at least 1; got {top_k}")
    config, tower = model.config, model.tower
    prompt_rows = len(prompt)
    if max_new_tokens < 1 or prompt_rows + max_new_tokens > config.text["max_position_embeddings"]:
        raise ValueError(f"max_new_tokens={max_new_tokens} does not fit after {prompt_rows} rows")

    caches = model.backbone.make_cache()
    tower_caches = tower.make_caches()
    rows = mx.array(prompt.astype(np.int32))
    for start in range(0, prompt_rows, PREFILL_CHUNK_SIZE):
        end = min(start + PREFILL_CHUNK_SIZE, prompt_rows)
        hidden = run_backbone(model, rows[start:end], caches)
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.forced(
                hidden[: paired_end - start],
                rows[start:paired_end],
                rows[start + 1 : paired_end + 1],
                tower_caches,
            )
        mx.eval(hidden, [c.state for c in caches], [c.state for c in tower_caches])

    hidden = hidden[-1:]
    frame = rows[-1:]
    history = rows[-RAS_WINDOW_FRAMES:, 0]
    produced, finish_reason = [], "length"
    for step in range(max_new_tokens):
        logits = model.backbone.lm_head(hidden)
        frame = tower.decode_step(
            logits, hidden, frame, history, tower_caches,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        history = mx.concatenate([history, frame[:, 0]])[-RAS_WINDOW_FRAMES:]
        if step + 1 < max_new_tokens:
            hidden = run_backbone(model, frame, caches)
            mx.async_eval(hidden, history)
        main, finite = mx.stack(
            [frame[0, 0], mx.isfinite(logits).all().astype(frame.dtype)]
        ).tolist()
        if not finite:
            raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
        if main == config.eos_token_id:
            finish_reason = "eos"
            break
        produced.append(frame)
        if watch is not None and watch.observe(frame[0, 1:]):
            finish_reason = "silence"
            break
    if not produced:
        return np.zeros((0, 1 + config.num_codebooks), dtype=np.int64), finish_reason
    return np.array(mx.concatenate(produced)).astype(np.int64), finish_reason


def generate_segment(model, text, *, references=(), previous_takes=(), temperature, top_k, top_p,
                     max_new_tokens, seed, watch=None):
    """One ``Drama3.generate(text, single_chunk=True, attempts=1)`` take, or None without frames.

    Raises the vendored ``PromptTooLongError`` before sampling, as the vendored
    runtime does, so callers keep Fish's history fallback order.
    """
    import mlx.core as mx
    from drama3_mlx.model import Segment
    from drama3_mlx.prompt import build_prompt

    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    mx.random.seed(seed)
    prompt = build_prompt(text, model.tokenizer, model.model.config,
                          [reference.text for reference in references],
                          [reference.codes for reference in references], list(previous_takes))
    started = time.perf_counter()
    frames, finish_reason = generate_frames_watched(
        model.model, prompt, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p, watch=watch)
    if not len(frames):
        return None
    codes = frames[:, 1:].T.copy()
    return Segment(text=text, codes=codes, audio=model.codec.decode(codes), finish_reason=finish_reason,
                   prompt_rows=len(prompt), seconds=time.perf_counter() - started)


def silence_watch(model, text):
    """The renderer's ``SilenceWatch`` decoding through the loaded model's codec."""
    import numpy as np
    import mlx.core as mx
    from .silence_stop import SilenceWatch

    def decode(frames):
        return model.codec.decode(np.array(mx.stack(list(frames), axis=1)))

    return SilenceWatch(text, decode, sample_rate=model.sample_rate)
