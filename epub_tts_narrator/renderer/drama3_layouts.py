"""Drama 3's streaming-continuation prompt layout, built beside the pinned runtime.

The vendored runtime builds only Fish's conversation layout, whose history
turns are earlier takes being revised. Drama 3 was also pre-trained on
consecutive chunks of one recording (technical report, appendix A, figure 10):

    <|im_start|>system
    streaming speech continuation with one-chunk lookahead split by <|endofturn|><|im_end|>
    <|im_start|>user
    {chunk}<|endofturn|>{next chunk}<|im_end|>
    <|im_start|>assistant
    <|voice|>{audio of chunk}<|im_end|>
    ...
    <|im_start|>user
    {new chunk}<|endofturn|><|im_end|>
    <|im_start|>assistant
    <|voice|>

The voice's reference recording is the first round, then the same history
takes, then the new line. Text is plain script words (no speaker tags or
bracketed controls), so it serves one voice without delivery controls. A
history-echo retry uses it; ordinary takes keep Fish's layout. Rows use the
runtime's own encoding: every literal tokenized on its own, audio rows as
``[c0 + semantic_start, c0..cN-1]``.
"""

STREAMING_INSTRUCTION = "streaming speech continuation with one-chunk lookahead split by <|endofturn|>"
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
TURN_SEPARATOR = "<|endofturn|>"


def streaming_segments(text, reference, history):
    """Prompt segments (strings and code arrays) for the streaming layout."""
    from .history_echo import plain_text
    rounds = [(plain_text(reference.text), reference.codes)]
    rounds += [(plain_text(turn_text), codes) for turn_text, codes in history]
    rounds = [(words, codes) for words, codes in rounds if words]
    target = plain_text(text)
    if not target:
        raise ValueError("The streaming history layout needs script words to speak")
    segments = ["<|im_start|>system\n", STREAMING_INSTRUCTION, MESSAGE_END]
    for index, (words, codes) in enumerate(rounds):
        following = rounds[index + 1][0] if index + 1 < len(rounds) else target
        segments += ["<|im_start|>user\n", f"{words}{TURN_SEPARATOR}{following}", MESSAGE_END,
                     GENERATION_START, codes, MESSAGE_END]
    segments += ["<|im_start|>user\n", f"{target}{TURN_SEPARATOR}", MESSAGE_END, GENERATION_START]
    return segments


def streaming_rows(text, reference, history, tokenizer, config):
    """int64 prompt rows for ``text`` after ``history`` in the streaming layout."""
    import numpy as np
    from drama3_mlx.prompt import MAX_PROMPT_ROWS, InvalidInputError, PromptTooLongError, validate_text
    num_codebooks = config.audio_decoder.num_codebooks
    codebook_size = config.audio_decoder.vocab_size
    rows = []
    for segment in streaming_segments(text, reference, history):
        if isinstance(segment, str):
            if segment not in {"<|im_start|>system\n", STREAMING_INSTRUCTION, MESSAGE_END,
                               "<|im_start|>user\n", GENERATION_START}:
                validate_text(segment.replace(TURN_SEPARATOR, " "), "streaming text")
            ids = np.array(tokenizer.encode(segment).ids, dtype=np.int64)
            block = np.zeros((len(ids), 1 + num_codebooks), dtype=np.int64)
            block[:, 0] = ids
            rows.append(block)
            continue
        codes = np.asarray(segment, dtype=np.int64)
        if codes.ndim != 2 or codes.shape[0] != num_codebooks or codes.shape[1] == 0:
            raise InvalidInputError(f"streaming codes must be [{num_codebooks}, frames >= 1]")
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise InvalidInputError(f"streaming codes must be in [0, {codebook_size})")
        rows.append(np.concatenate([codes[:1] + config.semantic_start_token_id, codes]).T)
    rows = np.concatenate(rows)
    if len(rows) > MAX_PROMPT_ROWS:
        raise PromptTooLongError(f"prompt has {len(rows)} rows; at most {MAX_PROMPT_ROWS}")
    return rows
