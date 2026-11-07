"""Fish's single-request Drama 3 decode loop with a per-frame stop check.

``generate_drama3_watched`` is Fish's ``generate_drama3`` (fish-speech
541566932abbaebcb0a40e562a1f846738383f38, the pinned Runpod runtime) step for
step: the same caches, CUDA graphs, prefill, sampler calls and random draws, so
a take that ends on its own returns exactly the frames Fish's function returns
for the same seed. The only addition is ``watch.observe`` after each kept frame
(see ``silence_stop``); when it returns True the loop ends with finish reason
``"silence"``. ``scripts/check_drama3_silence_stop.py`` checks the frame
equality on the worker; re-check it whenever the pinned Fish revision changes.

Fish and torch are imported inside the function; this module loads without them.
"""


def generate_drama3_watched(model, prompt, *, max_new_tokens, temperature, top_k, top_p, watch=None):
    """Frames after the prompt (CPU [frames, 1 + N], no EOS) and the finish reason."""
    import torch
    from fish_speech.models.text2semantic.fish_omni.generate import (
        DRAMA3_PREFILL_CHUNK_SIZE, Drama3DecodeGraphs, ensure_drama3_caches)

    if top_k < 1:
        raise ValueError(f"top_k must be at least 1; got {top_k}")
    backbone, tower = model.backbone, model.audio_decoder
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens
    if max_new_tokens < 1 or capacity > model.config.text.max_position_embeddings:
        raise ValueError(f"max_new_tokens={max_new_tokens} does not fit after {prompt_rows} prompt rows")
    graphs = Drama3DecodeGraphs.for_model(model, capacity)
    if graphs is None:
        ensure_drama3_caches(model, capacity)
    frames = torch.zeros((capacity, prompt.shape[1]), dtype=torch.long, device=tower.device)
    frames[:prompt_rows] = prompt.to(tower.device)
    positions = torch.arange(capacity, device=tower.device)

    for start in range(0, prompt_rows, DRAMA3_PREFILL_CHUNK_SIZE):
        end = min(start + DRAMA3_PREFILL_CHUNK_SIZE, prompt_rows)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
        hidden = backbone(inputs_embeds=embeddings[None], input_pos=positions[start:end])
        hidden = hidden[0].to(tower.device)
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.prefill(hidden[: paired_end - start], positions[start:paired_end],
                          frames[start:paired_end], frames[start + 1: paired_end + 1])

    hidden = hidden[-1:]
    produced, finish_reason = 0, "length"
    if graphs is not None:
        graphs.start(hidden, prompt_rows - 1, frames[:prompt_rows], temperature, top_k, top_p)
    for step in range(max_new_tokens):
        position = prompt_rows - 1 + step
        if graphs is not None:
            sampled, main = graphs.step(position)
        else:
            logits = backbone.compute_logits(hidden).to(tower.device)
            sampled = tower.decode_step(
                logits, hidden, positions[position: position + 1], frames[position: position + 1],
                frames[: position + 1, 0], temperature=temperature, top_k=top_k, top_p=top_p)
            main = sampled[0, 0].item()
        if main == model.config.eos_token_id:
            finish_reason = "eos"
            break
        frames[position + 1] = sampled[0]
        produced += 1
        # A view of the persistent frame buffer: later steps never overwrite it.
        if watch is not None and watch.observe(frames[position + 1, 1:]):
            finish_reason = "silence"
            break
        if graphs is None and step + 1 < max_new_tokens:
            embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
            hidden = backbone(inputs_embeds=embeddings[None],
                              input_pos=positions[position + 1: position + 2])[0].to(tower.device)
    return frames[prompt_rows: prompt_rows + produced].cpu(), finish_reason
