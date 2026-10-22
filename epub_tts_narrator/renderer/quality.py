"""Independent, fail-closed checks for generated speech.

The verifier receives only the audio path. Expected words are never supplied as
an ASR prompt or used as a replacement for an absent transcription.
"""

from __future__ import annotations

from array import array
import hashlib
from importlib import metadata
import math
from pathlib import Path
import re
import sys
import unicodedata
import wave

from .common import digest
from .phonetic_match import phonetic_equivalence


QUALITY_DEFAULTS = {
    "min_duration_seconds": 0.08,
    "max_duration_seconds": 120.0,
    "min_rms_dbfs": -55.0,
    "silence_threshold_dbfs": -60.0,
    "max_silence_fraction": 0.85,
    "max_internal_silence_seconds": 2.5,
    "max_clipped_fraction": 0.20,
    "max_wer": 0.0,
}
QUALITY_VERSION = 10
REFERENCE_VARIATION_VERSION = 6
_SAMPLE_RATE = 44100
_TIME_TOLERANCE = 0.05  # Whisper timestamps have coarser precision than samples.
_ASR_OPTIONS = {
    "task": "transcribe",
    "initial_prompt": None,
    "condition_on_previous_text": False,
    "temperature": 0.0,
    "word_timestamps": True,
    "verbose": None,
}
_REFERENCE_ASR_OPTIONS = {**_ASR_OPTIONS, "word_timestamps": False}


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_policy(policy: dict | None) -> dict:
    """Return a complete policy; reject typos and values that disable safeguards."""
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise ValueError("quality policy must be an object")
    unknown = set(policy) - QUALITY_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unknown quality settings: {', '.join(sorted(unknown))}")
    result = {**QUALITY_DEFAULTS, **policy}
    for key, value in result.items():
        if not _finite_number(value):
            raise ValueError(f"quality.{key} must be a finite number")
        if key.endswith("_dbfs"):
            if not -120 <= value <= 0:
                raise ValueError(f"quality.{key} must be between -120 and 0")
        elif key.endswith("_fraction") or key == "max_wer":
            if not 0 <= value <= 1:
                raise ValueError(f"quality.{key} must be between 0 and 1")
        elif value <= 0:
            raise ValueError(f"quality.{key} must be positive")
    if result["min_duration_seconds"] > result["max_duration_seconds"]:
        raise ValueError("quality minimum duration exceeds maximum duration")
    if result["silence_threshold_dbfs"] > result["min_rms_dbfs"]:
        raise ValueError("quality silence threshold must not exceed minimum RMS")
    return result


def _validate_purpose(purpose: str) -> str:
    if purpose not in ("speech", "reference"):
        raise ValueError("verification purpose must be 'speech' or 'reference'")
    return purpose


def validation_key(expected_text: str, verifier, policy: dict | None = None, *,
                   purpose: str = "speech") -> str:
    """Bind cached verification to its purpose, settings and expected words."""
    if not isinstance(expected_text, str):
        raise ValueError("expected_text must be a string")
    identity = {
        "policy_version": QUALITY_VERSION,
        "purpose": _validate_purpose(purpose),
        "quality": validate_policy(policy),
        "verifier": verifier.identity,
        "expected_text": expected_text,
    }
    if purpose == "reference":
        identity["reference_variation_version"] = REFERENCE_VARIATION_VERSION
    judge_identity = getattr(verifier, "reference_judge_identity", None)
    if judge_identity is not None:
        identity["text_judge"] = judge_identity
    return digest(identity)


_SMALL = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
_LEXEMES = re.compile(
    r"[-+]?\d+(?:[.,]\d+)*(?:st|nd|rd|th)?"
    r"|[^\W\d_]+(?:'[^\W\d_]+)*"
    r"|[%&+=*/$€£¥<>]"
    r"|\w+",
    re.UNICODE,
)
_SYMBOLS = {"%": ["percent"], "&": ["and"], "+": ["plus"], "=": ["equals"]}
_IDENTIFIER_TOKEN = re.compile(r"(?<![\w'’])[^\W_]+(?:-[^\W_]+)+(?![\w'’])", re.UNICODE)


def _lexeme_matches(text: str):
    """Preserve source offsets while separating clearly structured code parts.

    A hyphenated identifier needs both a mixed letter/digit component and a
    separate alphabetic component of at least two letters. Its hyphens separate
    code parts, rather than requesting the word 'minus'. Ambiguous forms such
    as x-34, x1-34 and purely numeric subtraction retain their signs. This
    affects verification only; source and synthesis text are never rewritten.
    """
    def separators(match):
        value = match.group()
        parts = value.split("-")
        has_mixed = any(any(char.isalpha() for char in part)
                        and any(char.isdigit() for char in part) for part in parts)
        has_word = any(len(part) >= 2 and part.isalpha() for part in parts)
        return value.replace("-", " ") if has_mixed and has_word else value

    # Replacing each hyphen with one space leaves all match offsets unchanged.
    return _LEXEMES.finditer(_IDENTIFIER_TOKEN.sub(separators, text))


def _integer_words(value: int) -> list[str]:
    if value < 20:
        return [_SMALL[value]]
    if value < 100:
        return [_TENS[value // 10]] + (_integer_words(value % 10) if value % 10 else [])
    if value < 1000:
        return [_SMALL[value // 100], "hundred"] + (_integer_words(value % 100) if value % 100 else [])
    return _integer_words(value // 1000) + ["thousand"] + (_integer_words(value % 1000) if value % 1000 else [])


def normalize_words(text: str) -> list[str]:
    """Conservative lexical normalization, with bounded cardinal equivalents.

    Apostrophes and meaningful arithmetic/currency symbols are retained.
    Leading-zero numbers, decimals, ordinals and integers >= one million stay
    literal: guessing their pronunciation could mask incorrect speech.
    """
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("‘", "'")
    words = []
    for match in _lexeme_matches(text):
        token = match.group()
        if token in _SYMBOLS:
            words.extend(_SYMBOLS[token])
            continue
        sign = token[:1] if token[:1] in ("+", "-") else ""
        digits = token[1:] if sign else token
        if re.fullmatch(r"[1-9]\d{0,2}(?:,\d{3})+", digits):
            digits = digits.replace(",", "")
        if re.fullmatch(r"0|[1-9][0-9]{0,5}", digits):
            if sign:
                words.append("minus" if sign == "-" else "plus")
            words.extend(_integer_words(int(digits)))
        else:
            words.append(token)
    return words


def compare_words(expected_text: str, actual_text: str) -> dict:
    """Word edit distance with a complete ordered, reviewable edit alignment."""
    expected, actual = normalize_words(expected_text), normalize_words(actual_text)
    return _compare_word_lists(expected, actual)


def _compare_word_lists(expected: list[str], actual: list[str]) -> dict:
    # Render chunks are bounded by the planner. Refuse pathological inputs
    # rather than allocating an unbounded quadratic matrix on a bad ASR result.
    if len(expected) > 2000 or len(actual) > 4000:
        raise ValueError("text exceeds per-chunk verification token limit")
    rows = [list(range(len(actual) + 1))]
    for i, left in enumerate(expected, 1):
        row = [i]
        for j, right in enumerate(actual, 1):
            row.append(min(rows[-1][j] + 1, row[-1] + 1, rows[-1][j - 1] + (left != right)))
        rows.append(row)
    i, j = len(expected), len(actual)
    edits = []
    counts = {"insertions": 0, "deletions": 0, "substitutions": 0}
    while i or j:
        if i and j and expected[i - 1] == actual[j - 1] and rows[i][j] == rows[i - 1][j - 1]:
            operation, previous_i, previous_j = "equal", i - 1, j - 1
        elif i and j and rows[i][j] == rows[i - 1][j - 1] + 1:
            operation, previous_i, previous_j = "substitute", i - 1, j - 1
            counts["substitutions"] += 1
        elif i and rows[i][j] == rows[i - 1][j] + 1:
            operation, previous_i, previous_j = "delete", i - 1, j
            counts["deletions"] += 1
        else:
            operation, previous_i, previous_j = "insert", i, j - 1
            counts["insertions"] += 1
        edits.append({
            "operation": operation,
            "expected_start": previous_i,
            "expected_end": i,
            "actual_start": previous_j,
            "actual_end": j,
            "expected": expected[previous_i:i],
            "actual": actual[previous_j:j],
        })
        i, j = previous_i, previous_j
    edits.reverse()
    error_count = sum(counts.values())
    return {
        **counts,
        "expected_word_count": len(expected),
        "actual_word_count": len(actual),
        "wer": error_count / len(expected) if expected else (0.0 if not actual else 1.0),
        "alignment": edits,
    }


_REFERENCE_COMPOUND = re.compile(
    r"(?<![\w'’‘\-\u2010\u2011])"
    r"[^\W\d_]{2,}(?:[-\u2010\u2011][^\W\d_]{2,})+"
    r"(?![\w'’‘\-\u2010\u2011])",
    re.UNICODE,
)
_NUMBER_COMPONENTS = frozenset(_SMALL + _TENS + [
    "hundred", "thousand", "million", "billion", "trillion", "minus", "plus",
])


def word_spacing_equivalence(expected_text: str, actual_text: str) -> dict | None:
    """Explain complete transcripts differing only in bounded word spacing.

    Join/split one alphabetic word and two or three adjacent pieces of at least
    two letters. The pieces must be separated by whitespace in the original
    text. Hyphens (e.g. re-sign/resign), apostrophes, numbers and changes to any
    letters still need ordinary verification. No insertion, deletion or
    substitution edge exists; even one other discrepancy rejects this shortcut.
    """
    def tokens(text):
        text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("‘", "'")
        words, spans = [], []
        for match in _lexeme_matches(text):
            normalized = normalize_words(match.group())
            words.extend(normalized)
            spans.extend([(match.start(), match.end(), match.group())] * len(normalized))
        return text, words, spans

    expected_source, expected, expected_spans = tokens(expected_text)
    actual_source, actual, actual_spans = tokens(actual_text)
    if len(expected) > 2000 or len(actual) > 4000:
        raise ValueError("text exceeds per-chunk verification token limit")
    if not expected or expected == actual or "".join(expected) != "".join(actual):
        return None

    def pieces(source, words, spans, start, size):
        end = start + size
        if end > len(words):
            return None
        selected = words[start:end]
        if any(not word.isalpha() or len(word) < 2 or word in _NUMBER_COMPONENTS
               for word in selected):
            return None
        if any(not span[2].isalpha() for span in spans[start:end]):
            return None
        if any(not source[spans[index - 1][1]:spans[index][0]].isspace()
               for index in range(start + 1, end)):
            return None
        return "".join(selected)

    i = j = 0
    changes = []
    while i < len(expected) and j < len(actual):
        if expected[i] == actual[j]:
            i, j = i + 1, j + 1
            continue
        left = pieces(expected_source, expected, expected_spans, i, 1)
        right = pieces(actual_source, actual, actual_spans, j, 1)
        match = None
        for size in (2, 3):
            if left is not None and left == pieces(actual_source, actual, actual_spans, j, size):
                match = (i + 1, j + size, "split")
                break
            if right is not None and right == pieces(expected_source, expected, expected_spans, i, size):
                match = (i + size, j + 1, "join")
                break
        if match is None:
            return None
        end_i, end_j, operation = match
        changes.append({"operation": operation, "expected_start": i, "expected_end": end_i,
                        "actual_start": j, "actual_end": end_j,
                        "expected": expected[i:end_i], "actual": actual[j:end_j]})
        i, j = end_i, end_j
    if i != len(expected) or j != len(actual):
        return None
    return {"method": "word_spacing_v1", "boundary_changes": len(changes), "alignment": changes}


def compare_reference_words(expected_text: str, actual_text: str, *,
                            source_text: str | None = None) -> dict:
    """Compare reference words using compounds explicitly written in the source.

    Only adjacent alphabetic components, each at least two letters, separated
    by a literal hyphen, Unicode hyphen or nonbreaking hyphen authorize joining.
    A recognizer cannot introduce new joining rules. Number-word compounds keep
    the existing cardinal normalization, and arbitrary spaces, identifiers,
    apostrophes, subtraction signs and other dashes do not create rules.

    This comparison is evidence for reference corroboration, not proof that
    different spellings have the same pronunciation. Raw word differences still
    require an independent recognizer. For two ASR transcripts, source_text must
    be the original requested script, not either recognizer's output.
    """
    source = expected_text if source_text is None else source_text
    expected, actual = normalize_words(expected_text), normalize_words(actual_text)
    if len(expected) > 2000 or len(actual) > 4000 or len(normalize_words(source)) > 2000:
        raise ValueError("text exceeds per-chunk verification token limit")
    compounds = set()
    # Extract before NFKC normalization so visually similar minus/dash symbols
    # cannot become authorized hyphens through compatibility normalization.
    for match in _REFERENCE_COMPOUND.finditer(source):
        components = re.split(r"[-\u2010\u2011]", match.group())
        component_words = [normalize_words(component) for component in components]
        if not all(component.isalpha() and len(words) == 1
                   for component, words in zip(components, component_words)):
            continue
        parts = tuple(words[0] for words in component_words)
        if not any(part in _NUMBER_COMPONENTS for part in parts):
            compounds.add(parts)
    ordered = sorted(compounds, key=lambda parts: (-len(parts), parts))

    def join_compounds(words):
        result, index = [], 0
        while index < len(words):
            for parts in ordered:
                if tuple(words[index:index + len(parts)]) == parts:
                    result.append("".join(parts))
                    index += len(parts)
                    break
            else:
                result.append(words[index])
                index += 1
        return result

    return {
        **_compare_word_lists(join_compounds(expected), join_compounds(actual)),
        "normalization": "source_explicit_compounds_v1",
        "source_compounds": [list(parts) for parts in ordered],
    }


def _audio_metrics(path: Path, policy: dict) -> dict:
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (
            1, 2, _SAMPLE_RATE, "NONE"
        ):
            raise ValueError("expected uncompressed PCM16 mono WAV at 44100 Hz")
        frame_count = audio.getnframes()
        duration = frame_count / _SAMPLE_RATE
        if duration > policy["max_duration_seconds"]:
            return {"duration_seconds": duration, "frames": frame_count, "sample_rate": _SAMPLE_RATE}
        raw = audio.readframes(frame_count)
        if len(raw) != frame_count * 2:
            raise ValueError("truncated WAV sample data")
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    square_sum = sum(value * value for value in samples)
    rms = math.sqrt(square_sum / len(samples)) / 32768 if samples else 0.0
    peak = max((abs(value) for value in samples), default=0) / 32768
    window = round(_SAMPLE_RATE * 0.025)
    threshold = (10 ** (policy["silence_threshold_dbfs"] / 20) * 32768) ** 2
    quiet = []
    silent_frames = 0
    for start in range(0, len(samples), window):
        frame = samples[start:start + window]
        is_quiet = sum(value * value for value in frame) / len(frame) < threshold
        quiet.append((is_quiet, len(frame)))
        if is_quiet:
            silent_frames += len(frame)
    voiced_windows = [i for i, (is_quiet, _) in enumerate(quiet) if not is_quiet]
    longest = current = 0
    if voiced_windows:
        for is_quiet, count in quiet[voiced_windows[0]:voiced_windows[-1] + 1]:
            current = current + count if is_quiet else 0
            longest = max(longest, current)
    return {
        "duration_seconds": duration,
        "frames": frame_count,
        "sample_rate": _SAMPLE_RATE,
        "rms_dbfs": 20 * math.log10(rms) if rms else None,
        "peak_dbfs": 20 * math.log10(peak) if peak else None,
        "silence_fraction": silent_frames / len(samples) if samples else 1.0,
        "longest_internal_silence_seconds": longest / _SAMPLE_RATE,
        "clipped_fraction": sum(abs(value) >= 32760 for value in samples) / len(samples) if samples else 0.0,
    }


class _AlignmentError(ValueError):
    """Optional ASR timing is unusable; transcript verification may continue."""


def _timestamp(value, label: str, duration: float) -> float:
    if not _finite_number(value) or value < 0 or value > duration + _TIME_TOLERANCE:
        raise _AlignmentError(f"{label} is not a finite timestamp within the audio")
    return min(float(value), duration)


def _segment_diagnostics(segment: dict) -> dict:
    diagnostic = {}
    for key in ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature"):
        if key in segment:
            value = segment[key]
            if not _finite_number(value):
                raise ValueError(f"ASR {key} must be finite")
            if key == "no_speech_prob" and not 0 <= value <= 1:
                raise ValueError("ASR no_speech_prob must be between zero and one")
            if key in ("compression_ratio", "temperature") and value < 0:
                raise ValueError(f"ASR {key} must be nonnegative")
            diagnostic[key] = value
    return diagnostic


def _checked_reference_transcript(result) -> tuple[str, list[dict], list[dict]]:
    """Check reference words without imposing an unused alignment contract.

    A cloning reference needs an accurate transcript, not playback timings.
    Ignore word and segment timing data entirely, including malformed timing
    supplied by a plugin whose transcription method also returns alignment.
    """
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ValueError("ASR result must contain a text string")
    if not normalize_words(result["text"]):
        raise ValueError("ASR produced no text")
    if not isinstance(result.get("segments"), list) or not result["segments"]:
        raise ValueError("ASR result must contain text segments")
    segment_texts, diagnostics = [], []
    for index, segment in enumerate(result["segments"]):
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError(f"ASR segment {index} lacks text")
        segment_texts.append(segment["text"])
        diagnostics.append(_segment_diagnostics(segment))
    if normalize_words(" ".join(segment_texts)) != normalize_words(result["text"]):
        raise ValueError("ASR segment text does not reconstruct the transcript")
    return result["text"], [], diagnostics


def _checked_transcript(result, duration: float) -> tuple[str, list[dict], list[dict]]:
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ValueError("ASR result must contain a text string")
    if not isinstance(result.get("segments"), list) or not result["segments"]:
        raise ValueError("ASR result must contain timed segments")
    words, diagnostics = [], []
    previous_segment_start = previous_word_end = 0.0
    segment_texts = []
    for index, segment in enumerate(result["segments"]):
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError(f"ASR segment {index} lacks text")
        start = _timestamp(segment.get("start"), f"segment {index} start", duration)
        end = _timestamp(segment.get("end"), f"segment {index} end", duration)
        if end < start or start + _TIME_TOLERANCE < previous_segment_start:
            raise _AlignmentError("ASR segments have reversed or unordered timestamps")
        previous_segment_start = start
        diagnostic = {"start": start, "end": end, **_segment_diagnostics(segment)}
        diagnostics.append(diagnostic)
        if not isinstance(segment.get("words"), list):
            raise _AlignmentError("ASR segment lacks word timestamps")
        word_texts = []
        for word in segment["words"]:
            if not isinstance(word, dict) or not isinstance(word.get("word"), str) or not word["word"].strip():
                raise _AlignmentError("ASR word lacks nonempty text")
            word_start = _timestamp(word.get("start"), "word start", duration)
            word_end = _timestamp(word.get("end"), "word end", duration)
            if word_end <= word_start:
                raise _AlignmentError("ASR word timestamps have zero or negative duration")
            if (word_start + _TIME_TOLERANCE < previous_word_end
                    or word_start + _TIME_TOLERANCE < start
                    or word_end > end + _TIME_TOLERANCE):
                raise _AlignmentError("ASR word timestamps overlap, are unordered, or leave their segment")
            checked = {"word": word["word"], "start": word_start, "end": word_end}
            if "probability" in word:
                probability = word["probability"]
                if not _finite_number(probability) or not 0 <= probability <= 1:
                    raise _AlignmentError("ASR word probability must be between zero and one")
                checked["probability"] = probability
            words.append(checked)
            word_texts.append(word["word"])
            previous_word_end = word_end
        if normalize_words(" ".join(word_texts)) != normalize_words(segment["text"]):
            raise _AlignmentError("ASR word text does not reconstruct its segment")
        segment_texts.append(segment["text"])
    if normalize_words(" ".join(segment_texts)) != normalize_words(result["text"]):
        raise ValueError("ASR segment text does not reconstruct the transcript")
    if not words:
        raise _AlignmentError("ASR produced no timed words")
    return result["text"], words, diagnostics


def _reference_variation_eligibility(metrics: dict) -> dict | None:
    """Bound reference edits without declaring word splits equivalent speech.

    A closed word and 2–3 adjacent alphabetic words with exactly the same
    letters may qualify for another recognizer. Each boundary change consumes
    one of the four allowed variations. Boundary-only changes preserve every
    letter in order and do not use raw WER, which counts one split as multiple
    errors. Any actual substitution retains the original 10% WER ceiling.
    Actual insertions/deletions have no permitted alignment edge.
    The final consensus comparisons never use these candidate boundary rules.
    """
    expected = [word for edit in metrics["alignment"] for word in edit["expected"]]
    actual = [word for edit in metrics["alignment"] for word in edit["actual"]]
    boundary_only = metrics["wer"] > .10
    if boundary_only and "".join(expected) != "".join(actual):
        return None
    limit = 4 if boundary_only else min(4, len(expected) // 10)

    def boundary(one, parts):
        return (one.isalpha() and "".join(parts) == one
                and all(part.isalpha() and len(part) >= 2
                        and part not in _NUMBER_COMPONENTS for part in parts))

    # At most four 1↔3 changes can move the alignment eight tokens off its
    # diagonal. A bounded dynamic program handles adjacent edits and repeats
    # without depending on arbitrary ties in the ordinary WER alignment.
    states = {(0, 0): (0, None, None)}

    def add(i, j, end_i, end_j, operation):
        cost = states[(i, j)][0] + (operation != "equal")
        if cost <= limit and ((end_i, end_j) not in states
                              or cost < states[(end_i, end_j)][0]):
            states[(end_i, end_j)] = (cost, (i, j), operation)

    for i in range(len(expected) + 1):
        for j in range(max(0, i - 2 * limit), min(len(actual), i + 2 * limit) + 1):
            if (i, j) not in states or i == len(expected) or j == len(actual):
                continue
            if expected[i] == actual[j]:
                add(i, j, i + 1, j + 1, "equal")
            elif not boundary_only:
                add(i, j, i + 1, j + 1, "substitute")
            for size in (2, 3):
                if j + size <= len(actual) and boundary(expected[i], actual[j:j + size]):
                    add(i, j, i + 1, j + size, "split")
                if i + size <= len(expected) and boundary(actual[j], expected[i:i + size]):
                    add(i, j, i + size, j + 1, "join")
    position = (len(expected), len(actual))
    if position not in states:
        return None
    changes = []
    while position != (0, 0):
        _, previous, operation = states[position]
        if operation != "equal":
            i, j = previous
            end_i, end_j = position
            changes.append({"operation": operation, "expected_start": i, "expected_end": end_i,
                            "actual_start": j, "actual_end": end_j,
                            "expected": expected[i:end_i], "actual": actual[j:end_j]})
        position = previous
    changes.reverse()
    return {"substitutions": sum(edit["operation"] == "substitute" for edit in changes),
            "boundary_changes": sum(edit["operation"] in {"split", "join"} for edit in changes),
            "variation_count": len(changes), "alignment": changes}


def _reference_stylization_eligibility(metrics: dict) -> dict | None:
    """Recognize patterned consonant stretching only to request corroboration.

    This does not rewrite the script or prove equivalent words/performance.
    It cannot be used by the final transcript comparisons or speech verifier.
    Ordinary double-letter edits alone lack the required stylization anchor.
    """
    expected = [word for edit in metrics["alignment"] for word in edit["expected"]]
    actual = [word for edit in metrics["alignment"] for word in edit["actual"]]
    if len(expected) != len(actual):
        return None
    consonants, distinct, base_words, edits = set(), set(), set(), []
    anchor = False
    for index, (source, heard) in enumerate(zip(expected, actual)):
        if source == heard:
            continue
        if (not re.fullmatch(r"[a-z]+", source) or not re.fullmatch(r"[a-z]+", heard)
                or source in _NUMBER_COMPONENTS or heard in _NUMBER_COMPONENTS):
            return None
        left = [match.group() for match in re.finditer(r"(.)\1*", source)]
        right = [match.group() for match in re.finditer(r"(.)\1*", heard)]
        if len(left) != len(right):
            return None
        reductions = []
        for run_index, (original, shortened) in enumerate(zip(left, right)):
            if original == shortened:
                continue
            letter = original[0]
            if (letter != shortened[0] or letter not in "bcdfghjklmnpqrstvwxyz"
                    or len(original) <= len(shortened)):
                return None
            consonants.add(letter)
            if len(consonants) > 1:
                return None
            # Initial "ss" marks the common hiss spelling. Do not infer a
            # general doubled-initial rule: "llama" is an ordinary word.
            anchor |= len(original) >= 3 or (run_index == 0 and letter == "s")
            reductions.append({"letter": letter, "source_count": len(original),
                               "heard_count": len(shortened), "run_index": run_index})
        distinct.add(source)
        base_words.add(heard)
        edits.append({"operation": "shorten_consonant_run", "word_index": index,
                      "expected": source, "actual": heard, "reductions": reductions})
    if len(distinct) < 3 or len(base_words) < 3 or not anchor:
        return None
    return {"method": "repeated_consonant_stylization_v1", "letter": next(iter(consonants)),
            "stylized_word_count": len(edits), "distinct_stylized_words": sorted(distinct),
            "distinct_heard_words": sorted(base_words),
            "alignment": edits,
            "performance_assessed": False}


def _valid_judge_model(model):
    return isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model) is not None


def _valid_judged_comparisons(record, report, secondary, *, partial=False, allow_model_change=False):
    """Bind every model score to the exact requested and recognized passages."""
    comparisons = record.get("comparisons")
    expected = report["expected_text"]
    groups = {}
    for label, heard in (("primary", report["transcript"]), ("secondary", secondary)):
        if heard is not None and compare_words(expected, heard)["wer"] != 0:
            groups.setdefault(heard, []).append(label)
    if (record.get("method") != "jev_spoken_passage" or not isinstance(comparisons, list)
            or not 1 <= len(comparisons) <= len(groups)
            or (not partial and len(comparisons) != len(groups))):
        return False
    for index, (item, (heard, labels)) in enumerate(zip(comparisons, groups.items())):
        if (not isinstance(item, dict) or set(item) != {
                "id", "requested_passage", "recognized_passage", "transcripts", "probability", "model", "policy_version"}
                or item["id"] != f"comparison_{index}"
                or item["requested_passage"] != expected or item["recognized_passage"] != heard
                or item["transcripts"] != labels
                or not _finite_number(item["probability"]) or not 0 <= item["probability"] <= 1
                or not _valid_judge_model(item["model"])
                or (item["model"] != record["model"] and not allow_model_change)
                or type(item["policy_version"]) is not int or item["policy_version"] != record["policy_version"]):
            return False
    return record["probabilities"]["roughly_same"] == max(item["probability"] for item in comparisons)


def _valid_unavailable_judgment(record, report, secondary):
    """Retain trustworthy partial judgments without treating them as approval."""
    error = record.get("error_code")
    if error is not None and (not isinstance(error, str) or error not in {
            "unavailable", "disabled", "invalid_request", "request_too_large", "invalid_response",
            "redirect_refused", "http_error", "transport_error", "invalid_comparison",
            "internal_error", "model_changed"}):
        return False
    usage = record.get("usage")
    if usage is not None and (not isinstance(usage, dict) or set(usage) != {"input_tokens", "output_tokens"}
                              or any(type(value) is not int or not 0 <= value <= 2 * 10**9 for value in usage.values())):
        return False
    probabilities, comparisons = record["probabilities"], record.get("comparisons", [])
    if probabilities == {} and record.get("model") is None:
        return comparisons == [] and error != "model_changed"
    if (error is None or error == "disabled" or set(probabilities) != {"roughly_same"}
            or not _finite_number(probabilities["roughly_same"])
            or not 0 <= probabilities["roughly_same"] <= 1
            or not _valid_judge_model(record.get("model"))
            or not _valid_judged_comparisons(record, report, secondary, partial=True, allow_model_change=error == "model_changed")):
        return False
    if error == "model_changed":
        return comparisons[0]["model"] == record["model"] and any(item["model"] != record["model"] for item in comparisons[1:])
    return True


def _judge_reference(verifier, report, secondary_transcript):
    """Require actual JEV approval of transcript differences for either purpose."""
    identity = getattr(verifier, "reference_judge_identity", None)
    if identity is None:
        return True
    unavailable = {"identity": identity, "decision": "unavailable", "probabilities": {},
                   "model": None, "error_code": "invalid_judge_record"}
    judge = getattr(verifier, "judge_reference", None)
    if not callable(judge):
        report["text_judge"] = dict(unavailable, error_code="judge_unavailable")
        return False
    try:
        record = judge(report["expected_text"], report["transcript"], secondary_transcript)
    except Exception:
        report["text_judge"] = dict(unavailable, error_code="judge_failed")
        return False
    if not isinstance(identity, dict) or not isinstance(record, dict) or record.get("identity") != identity:
        report["text_judge"] = unavailable
        return False
    record = {key: value for key, value in record.items() if key in {
        "identity", "decision", "probabilities", "model", "policy_version", "threshold", "usage", "error_code",
        "comparisons", "method", "questions"}}
    decision, probabilities = record.get("decision"), record.get("probabilities")
    threshold = record.get("threshold")
    if (not isinstance(decision, str) or decision not in {"equivalent", "different", "uncertain", "unavailable"}
            or not isinstance(probabilities, dict)
            or not _finite_number(threshold) or not .5 < threshold <= 1 or threshold != identity.get("threshold")
            or type(record.get("policy_version")) is not int or record["policy_version"] != identity.get("policy_version")):
        report["text_judge"] = unavailable
        return False
    if decision == "unavailable":
        report["text_judge"] = record if _valid_unavailable_judgment(record, report, secondary_transcript) else unavailable
        return False
    if (decision == "different" and record.get("method") == "spoken_text_eligibility"
            and record.get("error_code") in ("outside_spoken_text_limits", "invalid_comparison")
            and record.get("model") is None and probabilities == {} and record.get("comparisons") == []):
        report["text_judge"] = record
        report["text_judge_skipped"] = record["error_code"]
        return False
    if (set(probabilities) != {"roughly_same"}
            or any(not _finite_number(value) or not 0 <= value <= 1 for value in probabilities.values())
            or not _valid_judge_model(record.get("model"))
            or not _valid_judged_comparisons(record, report, secondary_transcript)):
        report["text_judge"] = unavailable
        return False
    report["text_judge"] = record
    return decision == "equivalent" and probabilities["roughly_same"] >= threshold


def text_confirmation_unavailable(report):
    """Provider outages must pause verification, never spend another TTS take."""
    if not isinstance(report, dict):
        return False
    judge = report.get("text_judge", {})
    reviews = report.get("phonetic_reviews", [])
    return ((isinstance(judge, dict) and judge.get("decision") == "unavailable")
            or (isinstance(reviews, list) and any(
                isinstance(row, dict) and row.get("decision") == "unavailable" for row in reviews)))


def _confirm_phonetics(verifier, report, transcript, source):
    """Ask the fallback only about bounded substitutions after JEV uncertainty."""
    if report.get("text_judge", {}).get("decision") != "uncertain":
        return False
    identity = getattr(verifier, "phonetic_review_identity", None)
    review = getattr(verifier, "review_phonetics", None)
    if not isinstance(identity, dict) or not callable(review):
        return False
    from .phonetic_review import substitution_pairs, validate_review
    expected = report["expected_text"]
    if substitution_pairs(expected, transcript) is None:
        return False
    unavailable = {"identity": identity, "decision": "unavailable", "source": source,
                   "error_code": "invalid_phonetic_review"}
    try:
        result = review(expected, transcript)
        accepted = validate_review(result, expected, transcript, identity)
        # Retain rejected/unavailable records from the trusted local reviewer;
        # only the complete strictly validated consensus can approve audio.
        if not isinstance(result, dict) or result.get("identity") != identity:
            result = unavailable
        elif result.get("decision") == "accepted" and not accepted:
            result = unavailable
        elif result.get("decision") not in {"accepted", "rejected", "unavailable"}:
            result = unavailable
    except Exception:
        result, accepted = unavailable, False
    report.setdefault("phonetic_reviews", []).append({**result, "source": source})
    return accepted


def _confirm_text_variation(path: Path, verifier, report: dict) -> bool:
    """Ask JEV about non-spacing mismatches; another recognizer can recover."""
    expected, primary = report["expected_text"], report["transcript"]

    def accept(transcript, method):
        report["verified_transcript"] = transcript
        report["text_confirmation_method"] = method
        if report["purpose"] == "reference":
            report.update(reference_text=transcript, reference_text_adjusted=transcript != expected)
        elif transcript != primary:
            report["words"] = []
            report["word_timing_status"] = "unavailable"
            report.setdefault("alignment_warnings", []).append(
                "Accepted secondary recognition; primary word timings are not used.")
        return True

    report["text_judge_source"] = "primary"
    if _judge_reference(verifier, report, None):
        return accept(primary, "jev_primary")
    if _confirm_phonetics(verifier, report, primary, "primary"):
        return accept(primary, "phonetic_primary")
    if text_confirmation_unavailable(report):
        return False
    decode = getattr(verifier, "corroborate_reference", None)
    if not callable(decode):
        return False
    corroboration = {"identity": getattr(verifier, "corroboration_identity", None),
                     "decision": "pending_judgment", "transcript": None}
    report["corroboration"] = corroboration
    try:
        # Expected text never goes to either recognizer.
        secondary, _, diagnostics = _checked_reference_transcript(decode(path))
        corroboration.update(transcript=secondary, asr_segments=diagnostics)
    except Exception:
        corroboration["decision"] = "verification_failed"
        return False
    if compare_words(primary, secondary)["wer"] == 0:
        corroboration["decision"] = "same_transcript"
        return False
    first_judgment = report.pop("text_judge")
    report["text_judge_history"] = [{"recognizer": "primary", "judgment": first_judgment}]
    if compare_words(expected, secondary)["wer"] == 0:
        report.pop("text_judge_source", None)
        corroboration["decision"] = "secondary_matches_requested"
        return accept(secondary, "secondary_exact")
    spacing = word_spacing_equivalence(expected, secondary)
    if spacing is not None:
        report.pop("text_judge_source", None)
        report["word_boundary_comparison"] = spacing
        corroboration["decision"] = "secondary_matches_requested_spacing"
        return accept(secondary, "secondary_word_spacing")
    phonetic = phonetic_equivalence(expected, secondary)
    if phonetic is not None:
        report.pop("text_judge_source", None)
        report["phonetic_comparison"] = phonetic
        corroboration["decision"] = "secondary_matches_requested_phonetically"
        return accept(secondary, "secondary_double_metaphone")
    secondary_report = {"expected_text": expected, "transcript": secondary}
    confirmed = _judge_reference(verifier, secondary_report, None)
    report["text_judge"] = secondary_report["text_judge"]
    report["text_judge_source"] = "secondary"
    corroboration["decision"] = "jev_confirmed_variations" if confirmed else "not_confirmed"
    if not confirmed and _confirm_phonetics(verifier, report, secondary, "secondary"):
        corroboration["decision"] = "phonetic_confirmed_variations"
        return accept(secondary, "phonetic_secondary")
    return accept(secondary, "jev_secondary") if confirmed else False


def _corroborate_reference(path: Path, verifier, report: dict) -> bool:
    """Resolve a small reference-script variation with an independent decoder.

    An audition script may be a book excerpt, but is not published book audio.
    If two decoders agree on a bounded variation, the actual words can become
    the cloning transcript.
    An independent exact match to the requested script can also resolve a small
    first-decoder discrepancy. Neither changes the rules for audiobook chunks.
    """
    metrics = report["reference_comparison"]
    corroboration = {
        "identity": getattr(verifier, "corroboration_identity", None),
        "decision": "mismatch_outside_reference_limits",
        "transcript": None,
    }
    report["corroboration"] = corroboration
    eligibility = (_reference_variation_eligibility(metrics)
                   or _reference_stylization_eligibility(metrics))
    if eligibility is None:
        return False
    corroboration["eligibility"] = eligibility
    decode = getattr(verifier, "corroborate_reference", None)
    if not callable(decode):
        corroboration["decision"] = "unavailable"
        return False
    try:
        # Audio alone: neither the requested script nor the first decoder's
        # words may bias the independent recognizer.
        result = decode(path)
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            corroboration["raw_transcript"] = result["text"]
        transcript, _, diagnostics = _checked_reference_transcript(result)
        corroboration["transcript"] = transcript
        corroboration["asr_segments"] = diagnostics
        primary_comparison = compare_reference_words(
            report["transcript"], transcript, source_text=report["expected_text"])
        requested_comparison = compare_reference_words(
            report["expected_text"], transcript)
        corroboration["primary_comparison"] = primary_comparison
        corroboration["requested_comparison"] = requested_comparison
    except Exception as exc:
        corroboration["decision"] = "verification_failed"
        corroboration["error"] = f"{type(exc).__name__}: {exc}"
        return False
    if primary_comparison["wer"] == 0:
        corroboration["decision"] = "agreed_reference_transcript"
        if requested_comparison["wer"] != 0 and not _judge_reference(verifier, report, transcript):
            return False
        report["reference_text"] = report["transcript"]
        report["reference_text_adjusted"] = True
        return True
    if requested_comparison["wer"] == 0:
        # This is an independent exact script match, not agreement between
        # decoders. Keep both transcripts and the primary mismatch diagnostics.
        corroboration["decision"] = "secondary_matches_requested"
        report["reference_text"] = report["expected_text"]
        report["reference_text_adjusted"] = False
        return True
    corroboration["decision"] = "disagreed"
    return False


def validate_audio(path, expected_text: str, verifier, policy: dict | None = None, *,
                   purpose: str = "speech") -> dict:
    """Check normalized PCM audio and an independently decoded transcript.

    Invalid policy/input text is a programming/configuration error. Bad audio,
    transcription failures and content mismatches produce a JSON-safe failure
    report, which can be retained alongside a rejected synthesis attempt.
    References do not require timing alignment that cloning never consumes.
    Exact word-spacing differences are accepted with their original audit.
    Other nonexact transcripts reach JEV when enabled. Hard acoustic checks
    still apply; unusable optional word timing
    is omitted with a warning rather than causing speech regeneration.
    """
    purpose = _validate_purpose(purpose)
    settings = validate_policy(policy)
    if not isinstance(expected_text, str):
        raise ValueError("expected_text must be a string")
    report = {"passed": False, "purpose": purpose, "expected_text": expected_text,
              "reasons": [], "transcript": None, "metrics": {}, "words": []}
    if not normalize_words(expected_text):
        report["reasons"].append("empty_expected_text")
        return report
    try:
        report["metrics"] = _audio_metrics(Path(path), settings)
    except (OSError, EOFError, ValueError, wave.Error) as exc:
        report["reasons"].append(f"invalid_audio: {exc}")
        return report
    metrics, reasons = report["metrics"], report["reasons"]
    duration = metrics["duration_seconds"]
    if duration < settings["min_duration_seconds"]:
        reasons.append("audio_too_short")
    if duration > settings["max_duration_seconds"]:
        reasons.append("audio_too_long")
        return report
    if metrics["rms_dbfs"] is None:
        reasons.append("silent_audio")
    elif metrics["rms_dbfs"] < settings["min_rms_dbfs"]:
        reasons.append("audio_too_quiet")
    if metrics["silence_fraction"] > settings["max_silence_fraction"]:
        reasons.append("excessive_silence")
    if metrics["longest_internal_silence_seconds"] > settings["max_internal_silence_seconds"]:
        reasons.append("internal_silence_stall")
    if metrics["clipped_fraction"] > settings["max_clipped_fraction"]:
        reasons.append("excessive_clipping")
    if reasons:
        return report
    try:
        # This call deliberately has no access to expected_text or any prompt.
        transcribe = getattr(verifier, "transcribe_reference", None) if purpose == "reference" else None
        if not callable(transcribe):
            transcribe = verifier.transcribe
        result = transcribe(Path(path))
        if isinstance(result, dict) and isinstance(result.get("text"), str):
            # Keep useful evidence even if an alignment/schema check fails;
            # this unvalidated text never substitutes for a passed transcript.
            report["raw_transcript"] = result["text"]
        transcript, words, diagnostics = _checked_reference_transcript(result)
        if purpose == "speech":
            try:
                transcript, words, diagnostics = _checked_transcript(result, duration)
                report["word_timing_status"] = "available"
            except _AlignmentError as exc:
                # Chapter assembly and source sync use actual clip boundaries.
                # Discard unusable optional word alignment, not valid speech.
                report["word_timing_status"] = "unavailable"
                report["alignment_warnings"] = [str(exc)]
        report["transcript"], report["words"] = transcript, words
        metrics["asr_segments"] = diagnostics
        metrics.update(compare_words(expected_text, transcript))
    except Exception as exc:
        # A verifier failure is never substituted with expected text.
        reasons.append(f"verification_failed: {type(exc).__name__}: {exc}")
        return report
    if purpose == "reference":
        report["reference_comparison"] = compare_reference_words(expected_text, transcript)
    spacing = word_spacing_equivalence(expected_text, transcript) if metrics["wer"] != 0 else None
    phonetic = (phonetic_equivalence(expected_text, transcript)
                if metrics["wer"] != 0 and spacing is None else None)
    if spacing is not None:
        report["word_boundary_comparison"] = spacing
        report["verified_transcript"] = transcript
        report["text_confirmation_method"] = "word_spacing"
        if purpose == "reference":
            report.update(reference_text=transcript, reference_text_adjusted=transcript != expected_text)
    elif phonetic is not None:
        report["phonetic_comparison"] = phonetic
        report["verified_transcript"] = transcript
        report["text_confirmation_method"] = "double_metaphone"
        if purpose == "reference":
            report.update(reference_text=transcript, reference_text_adjusted=transcript != expected_text)
    elif metrics["wer"] != 0 and getattr(verifier, "reference_judge_identity", None) is not None:
        if not _confirm_text_variation(Path(path), verifier, report):
            reasons.append("text_mismatch")
    elif purpose == "reference":
        if metrics["wer"] == 0:
            report["reference_text"] = expected_text
            report["reference_text_adjusted"] = False
        elif not _corroborate_reference(Path(path), verifier, report):
            # A permissive speech WER policy must never allow a reference to
            # be saved with a transcript that does not match its actual words.
            reasons.append("text_mismatch")
    elif metrics["wer"] > settings["max_wer"]:
        reasons.append("text_mismatch")
    report["passed"] = not reasons
    return report


def _hash_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


class MlxWhisperVerifier:
    """Local-only MLX Whisper adapter; model weights are fingerprinted once.

    A local directory is mandatory, so a misspelled repo name cannot trigger an
    implicit model download. Model/package/settings fingerprints belong in the
    render cache identity. Imports and model allocation wait until transcribe.
    """

    def __init__(self, model_path, language: str = "en"):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise ValueError(f"local ASR model directory does not exist: {self.model_path}")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("ASR language must be a nonempty language code")
        self.language = language
        model_files = [self.model_path / "config.json"]
        weight_files = sorted(self.model_path.glob("*.safetensors"))
        if (self.model_path / "weights.npz").is_file():
            weight_files.append(self.model_path / "weights.npz")
        if not model_files[0].is_file() or not weight_files:
            raise ValueError("local ASR model needs config.json and MLX weight files")
        model_files += weight_files
        self._model_file_state = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in model_files}
        try:
            version = metadata.version("mlx-whisper")
        except metadata.PackageNotFoundError:
            version = None
        packages = {}
        for name in ("mlx", "mlx-metal", "numpy", "tiktoken"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        self.identity = {
            "backend": "mlx-whisper",
            "package_version": version,
            "packages": packages,
            "model_path": str(self.model_path),
            "model_sha256": {path.name: _hash_file(path) for path in model_files},
            "language": language,
            "options": dict(_ASR_OPTIONS),
            "reference_options": dict(_REFERENCE_ASR_OPTIONS),
        }

    def transcribe(self, path) -> dict:
        return self._transcribe(path, _ASR_OPTIONS)

    def transcribe_reference(self, path) -> dict:
        """Decode reference speech without generating unused word alignment."""
        return self._transcribe(path, _REFERENCE_ASR_OPTIONS)

    def _transcribe(self, path, options: dict) -> dict:
        for filename, original_state in self._model_file_state.items():
            current = Path(filename).stat()
            if (current.st_size, current.st_mtime_ns) != original_state:
                raise RuntimeError("ASR model changed during this run; recreate the verifier")
        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError("mlx-whisper is required for local speech verification") from exc
        return mlx_whisper.transcribe(
            str(path),
            path_or_hf_repo=str(self.model_path),
            language=self.language,
            **options,
        )
