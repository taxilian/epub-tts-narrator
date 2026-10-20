"""Drama 3's documented scene prompt, restricted to one described speaker.

Figure 13 of https://arxiv.org/html/2610.18010v1#A2 places character
descriptions in the system turn and named dialogue in the user turn. The
pinned MLX runtime's ordinary TTS builder instead installs a fixed instruction
and prefixes speaker:0. Building these rows explicitly preserves the scene
format without modifying that runtime or its normal reference-cloning path.

Only one script line is supported: the runtime sampler cannot emit the audio
separator that the documented multi-line scene format requires.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
import unicodedata
from typing import Any


DESIGN_TEMPLATE_VERSION = "drama3-single-character-scene-v1"
MAX_PROMPT_ROWS = 8192
_SYSTEM_TEMPLATE = (
    "Scene: a quiet conversation in a room.\n"
    "Characters:\n"
    "- Speaker: {description}\n"
    "Voice every line of the script in order, with <|audio_separator|> "
    "between consecutive lines."
)
_SEGMENT_TEMPLATE = (
    "<|im_start|>system\n", _SYSTEM_TEMPLATE, "<|im_end|>\n",
    "<|im_start|>user\n", "Speaker: {text}", "<|im_end|>\n",
    "<|im_start|>assistant\n<|voice|>",
)
DESIGN_TEMPLATE_SHA256 = hashlib.sha256(
    json.dumps(_SEGMENT_TEMPLATE, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _plain_line(value: str, label: str, *, limit: int | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    if limit is not None and len(value) > limit:
        raise ValueError(f"{label} must contain at most {limit} characters")
    if any(char in value for char in "[]<>") or "|>" in value:
        raise ValueError(f"{label} contains Drama 3 control syntax")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"}
           and char not in "\n\r\t" for char in value):
        raise ValueError(f"{label} contains control characters")
    # A user-entered paragraph never becomes a second named script line or
    # character entry. Both fields occupy exactly one line in this template.
    return " ".join(value.split())


def validate_scene_fields(description: str, text: str) -> tuple[str, str]:
    return (
        _plain_line(description, "Voice description", limit=4000),
        _plain_line(text, "Sample text"),
    )


@dataclass(frozen=True)
class ScenePrompt:
    """Plain Python prompt rows keep construction testable without MLX."""

    rows: tuple[tuple[int, ...], ...]
    system: str
    user: str
    description: str
    text: str

    @property
    def audit(self) -> dict[str, Any]:
        return {
            "template_version": DESIGN_TEMPLATE_VERSION,
            "template_sha256": DESIGN_TEMPLATE_SHA256,
            "system": self.system,
            "user": self.user,
            "input_tokens": len(self.rows),
            "tokens_sha256": hashlib.sha256(
                json.dumps(self.rows, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "script_lines": 1,
            "speaker_tag_added": False,
        }


def build_scene_prompt(description: str, text: str, tokenizer: Any,
                       num_codebooks: int) -> ScenePrompt:
    description, text = validate_scene_fields(description, text)
    if type(num_codebooks) is not int or num_codebooks < 1:
        raise ValueError("Drama 3 must have at least one audio codebook")
    system = _SYSTEM_TEMPLATE.format(description=description)
    user = f"Speaker: {text}"
    segments = (
        "<|im_start|>system\n", system, "<|im_end|>\n",
        "<|im_start|>user\n", user, "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    )
    # Match drama3_mlx.prompt's literal boundaries. Tokenizing the concatenated
    # string produces different token IDs at these seams.
    rows = tuple(
        (token, *(0 for _ in range(num_codebooks)))
        for segment in segments for token in tokenizer.encode(segment).ids
    )
    if len(rows) > MAX_PROMPT_ROWS:
        raise ValueError(f"prompt has {len(rows)} rows; at most {MAX_PROMPT_ROWS}")
    return ScenePrompt(rows, system, user, description, text)


def generate_scene_voice(model: Any, description: str, text: str, *, seed: int,
                         temperature: float, top_p: float, top_k: int,
                         max_new_tokens: int) -> tuple[Any, dict[str, Any]]:
    """Return one runtime Segment and prompt audit; never retry or publish it."""
    import mlx.core as mx
    import numpy as np
    from drama3_mlx.generate import generate_frames
    from drama3_mlx.model import Segment

    config = model.model.config
    prompt = build_scene_prompt(
        description, text, model.tokenizer, config.audio_decoder.num_codebooks,
    )
    mx.random.seed(seed)
    started = time.perf_counter()
    frames, finish_reason = generate_frames(
        model.model, np.array(prompt.rows, dtype=np.int64),
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k,
    )
    if not len(frames):
        raise ValueError("Drama 3 returned no audio frames")
    codes = frames[:, 1:].T.copy()
    segment = Segment(
        text=text, codes=codes, audio=model.codec.decode(codes),
        finish_reason=finish_reason, prompt_rows=len(prompt.rows),
        seconds=time.perf_counter() - started,
    )
    return segment, prompt.audit
