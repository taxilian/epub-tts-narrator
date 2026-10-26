"""Compile validated portable performance plans at the model boundary.

Drama 3 markup is generated here, never accepted from book text. Controls are
requests, not an assertion that the resulting voice obeyed them acoustically.
"""

from __future__ import annotations

from copy import deepcopy
from bisect import bisect_left
import re
import unicodedata

from ..performance_plan import DELIVERY_TAGS, plain_direction, text_sha256, validate_performance, validate_phonemes


COMPILER_VERSION = "performance-compiler-v1"
PERFORMANCE_CAPABILITIES = {
    "drama3": {
        "version": 1, "scoped_delivery": sorted(DELIVERY_TAGS - {"style"}),
        "passage_direction": True, "scoped_style": "approximate_reset",
        "pronunciation_alphabets": ["cmu_arpabet", "respelling"], "inline_vocal_events": True,
        "standalone_vocal_events": False, "pause_duration": "approximate",
        "language_switching": False, "context_continuation": False,
    },
    "plain": {
        "version": 1, "scoped_delivery": [], "passage_direction": False,
        "scoped_style": False, "pronunciation_alphabets": [],
        "inline_vocal_events": False, "standalone_vocal_events": False,
        "pause_duration": False, "language_switching": False, "context_continuation": False,
    },
}
_SOUNDS = {"sigh": "sigh", "laugh": "laugh", "chuckle": "chuckle", "breath": "breath",
           "gasp": "gasp", "sob": "sob", "cough": "cough", "throat_clear": "clear throat"}
_PAIRED = DELIVERY_TAGS - {"style"}


def performance_kwargs(chunk, voice_id):
    """Optional engine arguments, without changing legacy plugin calls."""
    if "expressive_performance" not in chunk:
        return {}
    traits = chunk.get("expressive_traits", {})
    if not isinstance(traits, dict):
        raise ValueError("Expressive traits must be a voice mapping")
    return {"performance": chunk["expressive_performance"],
            "traits": plain_direction(traits.get(voice_id, ""), "Habitual voice traits")}


def trim_performance(text, record):
    """Project through the renderer's exact formatting removals, not alignment."""
    from .speech_text import speech_text_indices

    canonical = validate_performance(record, text)
    indices = speech_text_indices(text)
    prepared = "".join(text[i] for i in indices)
    if not indices:
        raise ValueError("Performance text has no speech after edge formatting removal")
    controls = []
    for original in canonical["controls"]:
        item = dict(original)
        if "offset" in item:
            item["offset"] = bisect_left(indices, item["offset"])
        else:
            a, b = bisect_left(indices, item["start"]), bisect_left(indices, item["end"])
            if a == b:
                raise ValueError("Performance control applies only to nonspoken edge formatting")
            if item["kind"] == "pronunciation" and b - a != item["end"] - item["start"]:
                raise ValueError("Edge formatting removal would cut through a pronunciation span")
            item.update(start=a, end=b, text=prepared[a:b])
        controls.append(item)
    return prepared, validate_performance({"version": 1, "text_sha256": text_sha256(prepared), "controls": controls}, prepared)


def _plain_text(text):
    if (not isinstance(text, str) or not text.strip()
            or any(c in text for c in "[]") or re.search(r"<[^>\n]*>", text)
            or any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} and c not in "\r\n\t" for c in text)):
        raise ValueError("Performance speech must be nonempty literal text without model control syntax")
    if not any(c.isalnum() for c in text):
        raise ValueError("Standalone nonverbal performance is not supported")


def validate_compiled_text(text):
    """Validate the trusted wire grammar, including paired delivery/CMU tags.

This checks structure, not source authorization. Callers must bind the request
to its validated plan; source text must pass the separate literal-text gate.
Speaker tokens are intentionally absent: dialogue adapters own their assembly.
"""
    if (not isinstance(text, str) or not text.strip() or len(text) > 1_000_000
            or any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} and c not in "\r\n\t" for c in text)):
        raise ValueError("Compiled performance must be bounded nonempty text")
    stack, index = [], 0
    while index < len(text):
        char = text[index]
        if char == "[":
            end = text.find("]", index + 1)
            if end < 0:
                raise ValueError("Compiled performance has an unterminated direction")
            plain_direction(text[index + 1:end], required=True)
            index = end + 1
        elif char == "]":
            raise ValueError("Compiled performance has an unmatched direction boundary")
        elif text.startswith("<|phoneme_start|>", index):
            start = index + len("<|phoneme_start|>")
            end = text.find("<|phoneme_end|>", start)
            if end < 0:
                raise ValueError("Compiled performance has an unterminated pronunciation")
            validate_phonemes(text[start:end])
            index = end + len("<|phoneme_end|>")
        elif char == "<" and (match := re.match(r"<([^>\n]*)>", text[index:])):
            tag = match.group(1)
            if tag in _PAIRED:
                stack.append(tag)
            elif tag.startswith("/") and stack and stack[-1] == tag[1:]:
                stack.pop()
            else:
                raise ValueError("Compiled performance contains unknown or unbalanced control tags")
            index += len(match.group(0))
        else:
            index += 1
    if stack:
        raise ValueError("Compiled performance has an unclosed paired direction")
    return text


def compile_performance(text, record, *, traits="", direction="", backend="drama3"):
    """Return engine text, unchanged expected words and explicit applied audit.

Use the existing adapter path for ``record is None``. Passing a record opts in
to this compiler and a separate cache identity; no legacy bytes are changed.
The plain backend explicitly reports every unsupported requested control.
"""
    _plain_text(text)
    canonical = validate_performance(record, text)
    traits = plain_direction(traits, "Habitual voice traits")
    direction = plain_direction(direction, "Passage direction")
    if not isinstance(backend, str) or backend not in PERFORMANCE_CAPABILITIES:
        raise ValueError("Performance compiler backend is unsupported")
    result = {
        "version": COMPILER_VERSION, "backend": backend, "text": text,
        "expected_text": text, "plan": canonical, "traits": traits,
        "direction": direction, "applied_controls": [], "warnings": [],
        "verification": {"lexical_text": text, "events": [], "preserve_leading_silence": False,
                         "preserve_trailing_silence": False, "requested_pause_ms": 0},
    }
    if backend == "plain":
        if traits or direction:
            result["warnings"].append("plain_backend_unsupported_direction")
        result["warnings"].extend(f"plain_backend_unsupported_{c['kind']}" for c in canonical["controls"])
        return result

    controls = canonical["controls"]
    starts, closes, replacements, events = {}, {}, {}, {}
    baseline = "; ".join(value for value in (traits, direction) if value) or "return to ordinary speaking delivery"
    for control in controls:
        item = deepcopy(control)
        kind = control["kind"]
        if kind == "delivery":
            a, b, tag = control["start"], control["end"], control["tag"]
            if tag == "style":
                starts[a] = f"[{control['direction']}] "
                if b < len(text):
                    closes[b] = f" [{baseline}] "
                    result["warnings"].append("scoped_style_reset_is_approximate")
                    item["scope_accuracy"] = "approximate_reset"
                else:
                    item["scope_accuracy"] = "starts_at_span"
            else:
                starts[a], closes[b] = f"<{tag}>", f"</{tag}>"
            result["applied_controls"].append(item)
        elif kind == "pronunciation":
            spoken = (control["replacement"] if control["alphabet"] == "respelling"
                      else f"<|phoneme_start|>{control['phonemes']}<|phoneme_end|>")
            replacements[control["start"]] = (control["end"], spoken)
            result["applied_controls"].append(item)
        else:
            offset = control["offset"]
            if kind == "pause":
                duration = format(control["duration_ms"] / 1000, ".3f").rstrip("0").rstrip(".")
                markup = f" [pause for {duration} seconds] "
                item["duration_accuracy"] = "approximate_model_request"
                result["warnings"].append("pause_duration_is_approximate")
                result["verification"]["requested_pause_ms"] += control["duration_ms"]
                result["verification"]["preserve_leading_silence"] |= offset == 0
                result["verification"]["preserve_trailing_silence"] |= offset == len(text)
            else:
                markup = f" [{_SOUNDS[control['sound']]}] "
            events.setdefault(offset, []).append(markup)
            result["verification"]["events"].append(deepcopy(control))
            result["applied_controls"].append(item)
    parts, index = [], 0
    while index <= len(text):
        # End the preceding scope, emit boundary events once, then start the
        # following scope. Events inside an open paired scope stay inside it.
        parts.append(closes.get(index, ""))
        parts.extend(events.get(index, []))
        parts.append(starts.get(index, ""))
        if index == len(text):
            break
        if index in replacements:
            index, markup = replacements[index]
            parts.append(markup)
        else:
            parts.append(text[index])
            index += 1
    engine_text = "".join(parts)
    # Keep traits independent: they must never turn a recognized whisper into
    # a generic style merely because the character has habitual speech traits.
    paired_direction = {"whispering": "whisper", "whispered": "whisper"}.get(direction.casefold(), direction.casefold())
    if paired_direction in _PAIRED:
        engine_text = f"<{paired_direction}>{engine_text}</{paired_direction}>"
    elif direction:
        engine_text = f"[{direction}] {engine_text}"
    if traits:
        engine_text = f"[{traits}] {engine_text}"
    result["text"] = validate_compiled_text(engine_text)
    result["warnings"] = list(dict.fromkeys(result["warnings"]))
    return result
